# 导入必要的包
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy, BaseModel, ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gym
from gym import spaces
import numpy as np
import itertools
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Callable
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class CLActor(BasePolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        num_tasks = 1,
        q_ewc_reg = 0.0,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        # CL Actor param
        self.num_tasks = num_tasks
        self.task_idx = 0
        self.q_ewc_reg = q_ewc_reg
        self.f_reg_term = None
        self.regularization_terms = {}


        action_dim = get_action_dim(self.action_space)
        state_dim = features_dim
        # latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        # self.latent_pi = nn.Sequential(*latent_pi_net)
        hidden_size = net_arch[0]
        self.latent_pi = nn.Sequential(nn.Linear(state_dim, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                    )
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            # self.mu = nn.Linear(last_layer_dim, action_dim)
            # self.log_std = nn.Linear(last_layer_dim, action_dim)
            self.mu = nn.ModuleDict()
            self.log_std =  nn.ModuleDict()
            for i in range(num_tasks):
                # self.mu[str(i)] = nn.Sequential(nn.Linear(hidden_size, action_dim))
                self.mu[str(i)] = nn.Linear(hidden_size, action_dim)
            for i in range(num_tasks):
                # self.log_std[str(i)] = nn.Sequential(nn.Linear(hidden_size, action_dim))
                self.log_std[str(i)] = nn.Linear(hidden_size, action_dim)

        self.heads = ["{0}.{1}.{2}".format("mu", a, b) for a, b in
                      itertools.product(range(self.num_tasks), ['weight', 'bias'])] + \
                     ["{0}.{1}.{2}".format("log_std", a, b) for a, b in
                      itertools.product(range(self.num_tasks), ['weight', 'bias'])]
        self.params = {n: p for n, p in self.named_parameters()}    # 需要先定义好网络结构再执行这一步

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)


    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor, task_idx=0) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        task_idx = int(task_idx)
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu[str(task_idx)](latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std[str(task_idx)](latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        idx = self.task_idx
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, int(idx))
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor, task_idx=0) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, int(task_idx))
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)

    def ewc_reg(self):
        reg_loss = 0
        if len(self.regularization_terms) > 0 and self.q_ewc_reg > 0:
            for i, reg_term in self.regularization_terms.items():  # diff items are diff task regs.
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
        print("actor_reg_loss", reg_loss)
        return self.q_ewc_reg * reg_loss

    def calculate_importance(self, state_all, q, alpha):
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)
        self.set_training_mode(False)
        N = list(state_all.size())[0]
        task_heads = ["{0}.{1}.{2}".format(a, int(self.task_idx), b) for a, b in itertools.product(['mu', 'log_std'], ['weight', 'bias'])]
        for i in range(N):
            s = state_all[i]
            q_ = q[i].view(1, -1)
            a, logprob = self.action_log_prob(s, self.task_idx)
            # a, logprob, _, _, _ = self.forward(s, evaluate=True)
            loss = (alpha * logprob - q_).mean()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            for n, p in importance.items():
                if self.params[n].grad is not None:
                    # if (str(n) in self.heads) == (str(n) in task_heads):
                    if (str(n) not in self.heads):
                        p += ((self.params[n].grad ** 2) * 1 / N)
        self.regularization_terms[int(self.task_idx)] = {'importance': importance, 'task_param': task_param}
        self.set_training_mode(True)

    def set_task(self, task_idx):
        self.task_idx = int(task_idx)

class CLContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        num_tasks = 1,
        q_ewc_reg = 0.0,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        self.num_tasks = num_tasks
        self.task_idx = 0
        self.q_ewc_reg = q_ewc_reg
        self.f_reg_term = None
        self.regularization_terms = {}


        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        self.q_networks_last = []
        hidden_size = net_arch[0]
        for idx in range(n_critics):
            # q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            # q_net = nn.Sequential(*q_net)
            q_net = nn.Sequential(nn.Linear(features_dim+action_dim, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU())
            q_net_last = nn.ModuleDict()
            for i in range(num_tasks):
                # q_net_last[str(i)] = nn.Sequential(nn.Linear(hidden_size, 1))
                q_net_last[str(i)] = nn.Linear(hidden_size, 1)
            self.add_module(f"qf{idx}", q_net)
            self.add_module(f"ql{idx}", q_net_last)
            self.q_networks.append(q_net)
            self.q_networks_last.append(q_net_last)

        self.heads = ["{0}.{1}.{2}".format("ql0", a, b) for a, b in
                      itertools.product(range(self.num_tasks), ['weight', 'bias'])] + \
                     ["{0}.{1}.{2}".format("ql1", a, b) for a, b in
                      itertools.product(range(self.num_tasks), ['weight', 'bias'])]
        self.params = {n: p for n, p in self.named_parameters()}

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        q_net_result = [q_net(qvalue_input) for q_net in self.q_networks]
        task_idx = int(self.task_idx)
        return tuple(q_net_last[str(task_idx)](q_net_result[i]) for i, q_net_last in enumerate(self.q_networks_last))

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        q_net_result = self.q_networks[0](th.cat([features, actions], dim=1))
        q_net_last = self.q_networks[0]
        task_idx = int(self.task_idx)
        return q_net_last[str(str(task_idx))](q_net_result)

    def ewc_reg(self):
        reg_loss = 0
        if len(self.regularization_terms) > 0 and self.q_ewc_reg > 0:
            for i, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
        print("critic_reg_loss", reg_loss)
        return self.q_ewc_reg * reg_loss

    def calculate_importance(self, state_all, action_all, q_target):
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()  # Backup the weight of current task
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # Zero initialize the importance matrix
        self.set_training_mode(False)
        N = list(state_all.size())[0]
        task_heads = ["{0}.{1}.{2}".format(a, int(self.task_idx), b) for a, b in itertools.product(['ql0', 'ql1'], ['weight', 'bias'])]
        for i in range(N):
            s = state_all[i]
            a = action_all[i]
            a = a.view(1,-1)
            q = q_target[i]
            q = q.view(-1,1)
            q1_current, q2_current = self.forward(s,a)
            loss = F.mse_loss(q1_current, q) + F.mse_loss(q2_current, q)
            self.optimizer.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    # if (str(n) in self.heads) == (str(n) in task_heads):
                    if (str(n) not in self.heads):
                        p += ((self.params[n].grad ** 2) * 1 / N)
        self.regularization_terms[int(self.task_idx)] = {'importance': importance, 'task_param': task_param}
        self.set_training_mode(True)

    def set_task(self, task_idx):
        self.task_idx = int(task_idx)

# 定义一个自定义的策略类，继承自BasePolicy
class CustomSACPolicy(BasePolicy):
    # 主体和SACPolicy一样，只改actor
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        num_tasks = 1,
        actor_q_ewc_reg = 0.0,
        critic_q_ewc_reg = 0.0,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.num_tasks = num_tasks
        self.actor_q_ewc_reg = actor_q_ewc_reg
        self.critic_q_ewc_reg = critic_q_ewc_reg
        # policy_aliases = {"CustomSACPolicy": (Box,)}
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1),
                                                    **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if
                                 "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CLActor:
        actor_kwargs = self.actor_kwargs.copy()
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["num_tasks"] = self.num_tasks
        actor_kwargs["q_ewc_reg"] = self.actor_q_ewc_reg
        return CLActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CLContinuousCritic:
        critic_kwargs = self.critic_kwargs.copy()
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["num_tasks"] = self.num_tasks
        critic_kwargs["q_ewc_reg"] = self.critic_q_ewc_reg
        return CLContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


# 定义一个自定义的SAC类，继承自SAC类
class CustomSAC(SAC):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "CustomSACPolicy": CustomSACPolicy,
    }

    def __init__(
        self,
        # Stablebaseline3
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # clsac
        num_tasks = 1,
        actor_q_ewc_reg=1,
        critic_q_ewc_reg=0.001,
        n_fisher_sample=10000,
    ):

        super(CustomSAC, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.num_tasks = num_tasks
        self.task_idx = 0
        self.tasks = {i: 0 for i in range(num_tasks)}
        self.ewc = True if actor_q_ewc_reg >= 0 else False  # can be 0 for ablation purposes.
        self.actor_q_ewc_reg = actor_q_ewc_reg
        self.critic_q_ewc_reg = critic_q_ewc_reg
        self.n_fisher_sample = n_fisher_sample
        # self.policy_kwargs = dict(num_tasks=num_tasks, q_ewc_reg=q_ewc_reg)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations, self.task_idx)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())
            self.ent_coef = ent_coef
            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations,self.task_idx)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            if self.ewc:
                critic_ewc_reg = self.critic.ewc_reg()
                critic_loss += critic_ewc_reg
                print(f"critic_loss:{critic_loss},critic_ewc_reg:{critic_ewc_reg}")
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            if self.ewc:
                actor_ewc_reg = self.actor.ewc_reg()
                actor_loss += actor_ewc_reg
                print(f"actor_loss:{actor_loss},actor_ewc_reg:{actor_ewc_reg}")
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        self.logger.record("train/episode_total_reward",  self.env.envs[0].episode_total_reward)
        self.logger.record("train/average_reward", self.env.envs[0].average_reward)
        self.logger.record("train/window_average_reward", self.env.envs[0].window_average_reward)
        self.logger.record("train/collision_rate", self.env.envs[0].collision_rate)
        self.logger.record("train/window_collision_rate", self.env.envs[0].window_collision_rate)
        self.logger.record("train/num_episode", self.env.envs[0].num_episode)


    def get_sample(self, sample_size):
        # if len(self.replay_buffer) > sample_size:
        #     data = self.replay_buffer.sample(sample_size, env=self._vec_normalize_env)
        # else:
        #     data = self.replay_buffer.get_all()
        replay_data = self.replay_buffer.sample(sample_size, env=self._vec_normalize_env)
        state_all = replay_data.observations
        next_state_all = replay_data.next_observations
        action_all = replay_data.actions
        reward_all = replay_data.rewards
        with th.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations,self.task_idx)
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - self.ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations, self.task_idx)
        q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        return state_all, action_all, target_q_values, min_qf_pi, self.ent_coef

    def set_task(self, task_idx: int, q_reg=False, memory=None):
        self.task_idx = int(task_idx)  # changing heads
        # 据说在保存和加载模型时使用了 json 模块或类似的方法，而 json 模块会将字典的键自动转换成字符串类型，以符合 JSON 格式的要求
        # 直接把 self.tasks 从 {0: 0, 1: 0} 变成了 {'0': 0, '1': 0}。。。下面重新变回去
        self.tasks = {int(k): v for k, v in self.tasks.items()}
        if q_reg:
            if self.ewc:
                print("Performing Q-EWC update")
                state_all, action_all, q_target, q, alpha = self.get_sample(self.n_fisher_sample)
                if self.actor_q_ewc_reg > 0:
                    self.critic.calculate_importance(state_all, action_all, q_target)
                    self.actor.calculate_importance(state_all, q, alpha)
                self.tasks[self.task_idx] += 1
        self.actor.set_task(int(self.task_idx))
        self.critic.set_task(int(self.task_idx))
        self.critic_target.set_task(int(self.task_idx))

    def get_task(self) -> int:
        return self.task_idx
