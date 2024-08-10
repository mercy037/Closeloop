from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from gym import spaces
import torch as th
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import GymEnv,Schedule
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.save_util import load_from_pkl
from torch.nn import functional as F
from stable_baselines3.common.utils import  polyak_update
from stable_baselines3.common.callbacks import BaseCallback


class ContinuousSAC(SAC):
    def __init__(self, q_ewc_reg=0.0):
        super().__init__()
        self.regularization_terms = {}
        self.q_ewc_reg = q_ewc_reg

    def train(self) -> None:
        # 添加您自己的训练步骤
        # super(CustomSAC, self).train()

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
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
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

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
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
        return self.q_ewc_reg * reg_loss

    def calculate_importance(self, state_all, q, alpha):
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)
        self.eval()
        N = list(state_all.size())[0]
        task_heads = ["{0}.{1}.{2}".format(a, int(self.task_idx), b) for a, b in itertools.product(['mean_out', 'log_std_out'], ['weight', 'bias'])]
        for i in range(N):
            s = state_all[i]
            s = s.unsqueeze(0)
            q_ = q[i].view(1, -1)
            a, logprob, _, _, _ = self.forward(s, evaluate=True)
            loss = (alpha * logprob - q_).mean()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            for n, p in importance.items():
                if self.params[n].grad is not None:
                    if (str(n) in self.heads) == (str(n) in task_heads):
                        p += ((self.params[n].grad ** 2) * 1 / N)
        self.regularization_terms[int(self.task_idx)] = {'importance': importance, 'task_param': task_param}