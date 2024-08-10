import random
import torch
import os
import itertools
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 0.1)
        if m.bias is not None:
            m.bias.data.fill_(0)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class BufferCache:
    def __init__(self, num_tasks):
        self._dict = {i: [] for i in range(num_tasks)}

    def set(self, i, mem):
        self._dict[int(i)] = mem

    def get(self, i):
        return self._dict[int(i)]

    def __str__(self):
        return str(["Task: {0}, size: {1}".format(t, len(l)) for t, l in self._dict.items()])

class ReplayPool:
    def __init__(self, capacity=1e4):
        self.capacity = int(capacity)
        self._memory = deque(maxlen=int(capacity))

    def push(self, transition: Transition):
        self._memory.append(transition)

    def sample(self, batch_size: int) -> Transition:
        transitions = random.sample(self._memory, batch_size)
        return Transition(*zip(*transitions))

    def get(self, start_idx: int, end_idx: int) -> Transition:
        transitions = list(itertools.islice(self._memory, start_idx, end_idx))
        return Transition(*zip(*transitions))

    def get_all(self) -> Transition:
        return self.get(0, len(self._memory))

    def __len__(self) -> int:
        return len(self._memory)

    def clear_pool(self):
        self._memory.clear()

    def get_list(self, batch_size: int) -> list:
        return list(random.sample(self._memory, int(batch_size)))

    def get_all_list(self) -> list:
        return list(itertools.islice(self._memory, 0, len(self._memory)))

    def set(self, s: list):
        for item in s:
            self.push(item)

def make_checkpoint_multi_task_car_following(agent, step_count, tag, counters):
    actor, critic, critic_target = agent.actor, agent.critic, agent.critic_target
    save_path = "checkpoints/model-{}-{}-{}.pt".format(step_count, tag, counters[2])
    if not os.path.isdir('checkpoints'):
        os.makedirs('checkpoints')
    torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'critic_target_state_dict': critic_target.state_dict(),
                'counters': counters,
                }, save_path)

def load_checkpoint_multi_task_car_following(agent, step_count, tag):
    save_path = "checkpoints/model-{}-{}.pt".format(step_count, tag)
    checkpoint = torch.load(save_path, map_location=device)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    counters = checkpoint['counters']
    cumulative_timestep, n_updates, i_episode, samples_number, task_number = counters

    return cumulative_timestep, n_updates, i_episode, samples_number, task_number