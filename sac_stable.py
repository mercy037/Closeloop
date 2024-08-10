import copy
import numpy as np
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from utils import init_params, ReplayPool, Transition

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, num_tasks=1, hidden_size=256):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        self.first = nn.Sequential(nn.Linear(state_dim, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                    )
        self.mean_layer = nn.ModuleDict()
        self.log_std_layer = nn.ModuleDict()
        for i in range(num_tasks):
            self.mean_layer[str(i)] = nn.Sequential(nn.Linear(hidden_size, action_dim))
        for i in range(num_tasks):
            self.log_std_layer[str(i)] = nn.Sequential(nn.Linear(hidden_size, action_dim))
        self.apply(init_params)

    def forward(self, s, task_idx=0):
        task_idx = int(task_idx)
        s = self.first(s)
        # print("state:{}, size{}".format(s, s.shape))
        # print("mean_tensor", self.mean_layer[str(task_idx)](s))
        mean = self.max_action * torch.tanh(self.mean_layer[str(task_idx)](s))  # 输出概率分布的均值mean
        # print("mean", mean)
        log_std = self.log_std_layer[str(task_idx)](s)
        # log_std = torch.clamp(log_std, -20, -4.5)
        log_std = torch.clamp(log_std, -2, -1)
        std = log_std.exp()  # 输出概率分布的标准差std
        return mean, std

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action, actor_lr=1e-3, num_tasks=1, hidden_size=256, q_ewc_reg=0.0):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.num_tasks = num_tasks
        self.network = ActorNetwork(state_dim, action_dim, max_action, self.num_tasks, hidden_size)
        self.task_idx = 0
        self.q_ewc_reg = q_ewc_reg
        self.f_reg_term = None
        self.regularization_terms = {}
        self.heads = ["{0}.{1}.{2}".format("mean_layer", a, b) for a, b in itertools.product(range(self.num_tasks), ['weight', 'bias'])] + \
                     ["{0}.{1}.{2}".format("log_std_layer", a, b) for a, b in itertools.product(range(self.num_tasks), ['weight', 'bias'])]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=actor_lr)
        self.params = {n: p for n, p in self.named_parameters()}

    def forward(self, s, choose_action=False, evaluate=False, head=None):
        idx = self.task_idx if head is None else head
        inputstate = torch.FloatTensor(s)
        mean, std = self.network.forward(inputstate, int(idx))
        dist = torch.distributions.Normal(mean, std)
        noise = torch.distributions.Normal(0, 0.1)
        if choose_action:
            action = dist.sample()
            action = torch.clamp(action, self.min_action, self.max_action)
            return action.detach().cpu().numpy()
        if evaluate:
            z = noise.sample()
            action = torch.tanh(mean + std * z)
            action = torch.clamp(action, self.min_action, self.max_action)
            action_logprob = dist.log_prob(mean + std * z) - torch.log(1 - action.pow(2) + 1e-6)
            return action, action_logprob, z, mean, std

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
        self.train()

    def set_task(self, task_idx):
        self.task_idx = int(task_idx)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_tasks, hidden_size=256):
        super(CriticNetwork, self).__init__()
        self.q1_first = nn.Sequential(nn.Linear(state_dim+action_dim, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU())
        self.q2_first = nn.Sequential(nn.Linear(state_dim+action_dim, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU())
        self.q1_last = nn.ModuleDict()
        self.q2_last = nn.ModuleDict()
        for i in range(num_tasks):
            self.q1_last[str(i)] = nn.Sequential(nn.Linear(hidden_size, 1))
        for i in range(num_tasks):
            self.q2_last[str(i)] = nn.Sequential(nn.Linear(hidden_size, 1))
        self.apply(init_params)

    def forward(self, s, a, task_idx=0):
        task_idx = int(task_idx)
        x = torch.cat((s, a), dim=1)
        q1 = self.q1_first(x)
        q1 = self.q1_last[str(task_idx)](q1)
        q2 = self.q2_first(x)
        q2 = self.q2_last[str(task_idx)](q2)
        return q1, q2

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_tasks, critic_lr=3e-3, hidden_size=256, q_ewc_reg=0.0):
        super(Critic, self).__init__()
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.network = CriticNetwork(state_dim, action_dim, self.num_tasks, hidden_size)
        self.task_idx = 0
        self.q_ewc_reg = q_ewc_reg
        self.f_reg_term = None
        self.regularization_terms = {}
        self.heads = ["{0}.{1}.{2}".format("q1_last", a, b) for a, b in itertools.product(range(self.num_tasks), ['weight', 'bias'])] + \
                     ["{0}.{1}.{2}".format("q2_last", a, b) for a, b in itertools.product(range(self.num_tasks), ['weight', 'bias'])]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=critic_lr, eps=1e-5)
        self.params = {n: p for n, p in self.named_parameters()}

    def forward(self, s, a, head=None):
        idx = self.task_idx if head is None else head
        q1, q2 = self.network.forward(s, a, int(idx))
        return q1, q2

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
        return self.q_ewc_reg * reg_loss

    def calculate_importance(self, state_all, action_all, q_target):
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()  # Backup the weight of current task
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # Zero initialize the importance matrix
        self.eval()
        N = list(state_all.size())[0]
        task_heads = ["{0}.{1}.{2}".format(a, int(self.task_idx), b) for a, b in itertools.product(['q1_out', 'q2_out'], ['weight', 'bias'])]
        for i in range(N):
            s = state_all[i]
            s = s.unsqueeze(0)
            a = action_all[i]
            a = a.view(-1,1)
            q = q_target[i]
            q = q.view(-1,1)
            q1_current, q2_current = self.forward(s,a)
            loss = F.mse_loss(q1_current, q) + F.mse_loss(q2_current, q)
            self.optimizer.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    if (str(n) in self.heads) == (str(n) in task_heads):
                        p += ((self.params[n].grad ** 2) * 1 / N)
        self.regularization_terms[int(self.task_idx)] = {'importance': importance, 'task_param': task_param}
        self.train()

    def set_task(self, task_idx):
        self.task_idx = int(task_idx)

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, min_action, num_tasks, hidden_size, actor_lr, critic_lr,
                 q_ewc_reg=0.0, gamma=0.9, q_lr=2e-3, batch_size=128, tau=1e-2,adaptive_alpha=True, pool_size=10000):
        self.gamma = gamma
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.tau = tau
        self.task_idx = 0
        self.tasks = {i: 0 for i in range(num_tasks)}
        self.ewc = True if q_ewc_reg >= 0 else False  # can be 0 for ablation purposes.
        self.q_ewc_reg = q_ewc_reg
        self.adaptive_alpha = adaptive_alpha
        if self.adaptive_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.alpha = 0.2
        torch.manual_seed(1)  # 设置随机种子
        self.actor = Actor(state_dim, action_dim, max_action, min_action, actor_lr, num_tasks, hidden_size, q_ewc_reg)
        self.critic = Critic(state_dim, action_dim, num_tasks, critic_lr, hidden_size, q_ewc_reg)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_pool = ReplayPool(capacity=int(pool_size))

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.actor.train()
        self.critic.train()
        self.critic_target.train()

    def choose_action(self, s, eval=False):
        if eval:
            evaluate = True
            action = self.actor(s, evaluate)
        else:
            choose_action = True
            action = self.actor(s, choose_action)
        return action

    def optimise(self):
        samples = self.replay_pool.sample(self.batch_size)
        state_batch = torch.FloatTensor(samples.state)
        action_batch = torch.FloatTensor(samples.action)
        reward_batch = torch.FloatTensor(samples.reward).unsqueeze(1)
        next_state_batch = torch.FloatTensor(samples.next_state)

        current_Q1, current_Q2 = self.critic(state_batch,action_batch)
        with torch.no_grad():
            next_action_batch, log_prob_, z, mean, log_std = self.actor(next_state_batch, evaluate=True)
            target_Q1, target_Q2 = self.critic_target(next_state_batch,  next_action_batch)
            target_Q = reward_batch + self.gamma * (torch.min(target_Q1,target_Q2)-self.alpha*log_prob_)  # Compute target Q
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)  # Compute critic loss
        if self.ewc:
            critic_loss += self.critic.ewc_reg()
        self.critic_optimizer.zero_grad()  # Optimize the critic
        critic_loss.backward()
        self.critic_optimizer.step()

        a, log_prob, _, _, _ = self.actor(state_batch, evaluate=True)
        Q1, Q2 = self.critic(state_batch, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_prob - Q).mean()  # Compute actor loss
        if self.ewc:
            actor_loss += self.actor.ewc_reg()
        self.actor_optimizer.zero_grad()  # Optimize the actor
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.adaptive_alpha:  # Update alpha
            alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):  # Softly update target network
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return self.alpha

    def set_task(self, task_idx, q_reg=False):
        return None

    def get_task(self):
        return None

class SACAgentOWL(SACAgent):
    def __init__(self, state_dim, action_dim, max_action, min_action, num_tasks, hidden_size, actor_lr, critic_lr, q_ewc_reg,
                 n_fisher_sample=10000, gamma=0.9, q_lr=2e-3, batch_size=128, tau=1e-2,adaptive_alpha=True, pool_size=10000):

        super(SACAgentOWL, self).__init__(state_dim=state_dim, action_dim=action_dim, max_action=max_action, min_action=min_action,
                                          num_tasks=num_tasks, hidden_size=hidden_size, actor_lr=actor_lr, critic_lr=critic_lr,
                                          q_ewc_reg=q_ewc_reg, gamma=gamma, q_lr=q_lr, batch_size=batch_size, tau=tau,
                                          adaptive_alpha=adaptive_alpha, pool_size=pool_size)

        self.n_fisher_sample = n_fisher_sample

    def get_sample(self, sample_size):
        if len(self.replay_pool) > sample_size:
            data = self.replay_pool.sample(sample_size)
        else:
            data = self.replay_pool.get_all()
        state_all = torch.FloatTensor(data.state)
        next_state_all = torch.FloatTensor(data.next_state)
        action_all = torch.FloatTensor(data.action)
        reward_all = torch.FloatTensor(data.reward).unsqueeze(1)
        with torch.no_grad():
            next_action_all, log_prob_, z, mean, log_std = self.actor(next_state_all, evaluate=True)
            q1_target, q2_target = self.critic_target(next_state_all, next_action_all)
            q_target = reward_all + self.gamma * (torch.min(q1_target, q2_target) - self.alpha * log_prob_)
        a, log_prob, _, _, _ = self.actor(state_all, evaluate=True)
        q1, q2 = self.critic(state_all, a)
        q = torch.min(q1, q2)
        return state_all, action_all, q_target, q, self.alpha

    def set_task(self, task_idx: int, q_reg=False, memory=None):
        self.task_idx = int(task_idx)  # changing heads
        if q_reg:
            if self.ewc:
                print("Performing Q-EWC update")
                state_all, action_all, q_target, q, alpha = self.get_sample(self.n_fisher_sample)
                if self.q_ewc_reg > 0:
                    self.critic.calculate_importance(state_all, action_all, q_target)
                    self.actor.calculate_importance(state_all, q, alpha)
                self.tasks[self.task_idx] += 1
        self.actor.set_task(int(self.task_idx))
        self.critic.set_task(int(self.task_idx))
        self.critic_target.set_task(int(self.task_idx))

    def get_task(self) -> int:
        return self.task_idx















