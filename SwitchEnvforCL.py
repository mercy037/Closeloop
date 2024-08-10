
from stable_baselines3.common.callbacks import BaseCallback
import os

class SwitchEnvCallback(BaseCallback):
    def __init__(self, env_list, switch_freq, verbose=0):
        super(SwitchEnvCallback, self).__init__(verbose)
        self.env_list = env_list    # a list of environments to switch
        self.switch_freq = switch_freq  # the frequency of switching in steps
        self.env_index = 0  # the index of the current environment
        self.num_steps = 0 # the number of steps taken so far
        self.task_idx = 0

    def _on_step(self) -> bool:

        self.num_steps += 1
        # check if it is time to switch the environment
        # print("current env:", self.env_index)
        if  self.num_steps % self.switch_freq[self.task_idx] == 0:
            # # 冻结训练完成任务的策略头网络
            # actor_trained_head_name = [f'mu.{self.task_idx}.weight', f'mu.{self.task_idx}.bias',
            #                            f'log_std.{self.task_idx}.weight', f'log_std.{self.task_idx}.bias']
            # critic_trained_head_name = [f'ql0.{self.task_idx}.weight', f'ql0.{self.task_idx}.bias',
            #                             f'ql1.{self.task_idx}.weight', f'ql1.{self.task_idx}.bias']
            # for head_name in actor_trained_head_name:
            #     self.model.actor.params[head_name].requires_grad = False
            # for head_name in critic_trained_head_name:
            #     self.model.critic.params[head_name].requires_grad = False
            #     self.model.critic_target.params[head_name].requires_grad = False

            # select the next environment in the list
            self.env_index = (self.env_index + 1) % len(self.env_list)
            task_number = self.env_index    # task number 表示即将切换进行训练的任务编号
            self.model.set_task(task_number, q_reg=True)
            # batch_size = self.model.learning_starts
            batch_size = 1
            batch = self.model.replay_buffer.sample(batch_size,
                                                    env=self.model._vec_normalize_env)  # 从replay_buffer中采样一批数据
            # 切换环境后，需要清空replay_buffer，但是stable_baseline没有清空函数，只能重置了
            self.model.replay_buffer = self.model.replay_buffer_class(
                self.model.buffer_size,
                self.model.observation_space,
                self.model.action_space,
                device=self.model.device,
                n_envs=self.model.n_envs,
                optimize_memory_usage=self.model.optimize_memory_usage,
                **self.model.replay_buffer_kwargs,
            )
            # replay_buffer重置后 replay_buffer.pos=0 而 self.model.num_timesteps > self.model.learning_starts
            # 此时会直接开始训练而不是先收集100个场景，为避免报错，先填充一组数据到新的replay_buffer中使replay_buffer.pos>0
            # replay_buffer.add需要六个输入，对应obs, next_obs, action, reward, done, infos，
            # 前五个可以从batch中按对应顺序挑出来，并将tensor转移到cpu上
            # infos只要是一个由字典构成的列表就可以，内容无所谓。
            infos_init = {'episode_total_reward': 0, 'average_reward': 0, 'window_average_reward': 0,
                          'collision_rate': 0, 'window_collision_rate': 0, 'num_episode': 0,
                          'info': {'reserved': 0, 'DataInfo': [0.0], 'Collision': False}}
            self.model.replay_buffer.add(batch[0][0].cpu(), batch[2][0].cpu(), batch[1][0].cpu(), batch[4][0].cpu(),
                                         batch[3][0].cpu(), [infos_init])
            # self.model.num_timesteps = 0
            # self.model.env.envs[0].num_episode = 0
            # self.model.env.envs[0].num_collision = 0
            # self.model.env.envs[0].total_reward = 0
            # self.model.env.envs[0].reward_window = []
            # self.model.env.envs[0].collision_window = []

            # get the new environment
            new_env = self.env_list[self.env_index]
            print(f"Switching to environment {new_env}")
            # set the new environment for the model
            self.model.set_env(new_env)
            self.model.env.envs[0].reset()
            self.task_idx += 1  # task_idx 表示当前训练的任务数量
        return True
