import os
import gym
import collections
import copy
import numpy as np
import argparse
import Carla_gym
from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from datetime import datetime
from sac import SACAgent, SACAgentOWL
from evaluate_agent import evaluate_agent_oracle, evaluate_agent_bandits
from utils import Transition, make_checkpoint_multi_task_car_following, load_checkpoint_multi_task_car_following
from torch.utils.tensorboard import SummaryWriter
from utils import BufferCache
import torch
def train_agent_model_free_multi_task_car_following(agent, envs, params):
    bandit_eval = params['bandits']
    episode_max_steps = params['episode_max_steps']
    log_interval = params['log_interval']
    max_task_frames = params['max_task_frames']
    num_tasks = params['num_tasks']
    num_repeats = params['num_task_repeats']
    n_episodes = params['n_episodes']
    n_collect_steps = params['n_collect_steps']
    owl = params['owl']
    save_model = params['save_model']
    tag = params['tag']
    tag = ''
    buffer_warm_start = params['buffer_warm_start']
    warm_start_size = params['buffer_warm_start_size']

    assert n_collect_steps >= agent.batch_size  # We must initially collect as many steps as the batch size!

    cumulative_timestep = 0
    n_updates = 0
    i_episode = 0
    samples_number = 0
    task_number = 0
    i_tasks_prev = 0
    episode_reward = collections.deque(maxlen=1)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(params['logdir'], current_time + '_' + tag)
    writer = SummaryWriter(log_dir=log_dir)
    cache = BufferCache(num_tasks)

    # cumulative_timestep, n_updates, i_episode, samples_number, task_number = load_checkpoint_multi_task_car_following(
    #     agent, 'Dec02_09-41-26', 'sac_ewc_car_following_env_0')
    n_updates = 0
    samples_number = 0
    while True:
        time_step = 0
        i_task = cumulative_timestep // max_task_frames  # 取整
        i_episode += 1
        if i_episode % 1000 == 0:  # Condition to terminate, also let's cache agent and envs for plotting later
            if save_model:
                current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                make_checkpoint_multi_task_car_following(agent, current_time, tag,
                                                         [cumulative_timestep, n_updates, i_episode, samples_number,
                                                          task_number])

        # print("time_step:{},i_episode:{},cumulative_timestep:{},i_task:{},max_task_frames:{},task_number:{}".format(time_step,i_episode,cumulative_timestep,
        #                                                                                                             i_task,max_task_frames,task_number))
        # if i_task != i_tasks_prev:
        #     i_tasks_prev = i_task
        #     task_number = i_task % num_tasks  # 取余数
        #
        #     # if i_task >= num_tasks*num_repeats:  # Condition to terminate, also let's cache agent and envs for plotting later
        #     #     if save_model:
        #     #         print("save")
        #     #         make_checkpoint_multi_task_car_following(agent, current_time, tag,
        #     #                                                  [cumulative_timestep, n_updates, i_episode, samples_number, task_number])
        #     #     return
        #
        #     # owl oracle used to tell what task to train on
        #     # Change heads and warm start buffer or warm start the epsilon greedy strategy
        #     if owl:
        #         print("Manual switch")
        #         if buffer_warm_start:  # sample data from the buffer
        #             if len(agent.replay_pool) > warm_start_size:
        #                 memory = agent.replay_pool.get_list(warm_start_size)
        #             else:
        #                 memory = agent.replay_pool.get_all_list()
        #             cache.set((task_number - 1) % num_tasks, memory)
        #         else:
        #             memory = None
        #         agent.set_task(task_number, q_reg=True, memory=memory)
        #         agent.replay_pool.clear_pool()
        #         if buffer_warm_start and i_task > 1:  # Warm start the buffer with data from the Cache
        #             memory = cache.get(task_number)
        #             agent.replay_pool.set(memory)
        #         samples_number = 0

        env = envs[int(task_number)]
        state = env.reset()
        done = False
        episode_score = 0
        while not done:
            agent.eval()
            action = agent.choose_action(state)
            agent.train()
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state)  # 将state转换为NumPy数组
            state_normalized = (next_state - next_state.min()) / (next_state.max() - next_state.min())
            next_state = state_normalized
            # writer.add_scalar('Info/相对距离', state[0], cumulative_timestep)
            # writer.add_scalar('Info/后车速度', state[2], cumulative_timestep)
            # writer.add_scalar('Info/collision', info[0], cumulative_timestep)
            # writer.add_scalar('Info/n_step', info[1], cumulative_timestep)
            # writer.add_scalar('Info/ttc', info[2], cumulative_timestep)
            # writer.add_scalar('Info/hdw', info[3], cumulative_timestep)
            # writer.add_scalar('Info/jerk', info[4], cumulative_timestep)
            # writer.add_scalar('Info/前车s', info[5], cumulative_timestep)
            # writer.add_scalar('Info/后车s', info[6], cumulative_timestep)
            # writer.add_scalar('Info/前车速度', info[7], cumulative_timestep)

            time_step += 1
            cumulative_timestep += 1
            episode_score += reward
            agent.replay_pool.push(Transition(state, action, reward, next_state))
            samples_number += 1
            state = next_state
            # print(state)

            alpha_ = agent.optimise()
            n_updates += 1
            if (time_step >= 1000):
                done = True
            if done:
                mean_episode_score = episode_score / time_step
                print("episode:{}, mean episode score:{}, cumulative_timestep:{}".format(
                    i_episode, mean_episode_score, cumulative_timestep))
                episode_reward.append(mean_episode_score)
"""
            if cumulative_timestep % log_interval == 0:
                # writer.add_scalar('Params/alpha', alpha_, cumulative_timestep)
                episodes_return_mean, episode_return,  = evaluate_agent_oracle(envs, agent, episode_max_steps, n_episodes=n_episodes)

                if bandit_eval:
                    episodes_return_mean_bandit, episode_return_bandit, corrects_bandit, bandit_logging = evaluate_agent_bandits(
                        envs, agent, episode_max_steps, params['bandit_loss'], params['greedy_bandit'],
                        n_episodes=n_episodes, n_arms=num_tasks, lr=params['bandit_lr'], decay=params['bandit_decay'],
                        epsilon=params['bandit_epsilon'], bandit_step=params['bandit_step'])

                writer.add_scalar('Reward/Task', task_number, cumulative_timestep)  # 记录当前在哪个任务上训练

                for i in range(num_tasks):
                    Returns = episodes_return_mean[i]
                    Returns_unmean = episode_return[i]
                    writer.add_scalar('Reward/ORCL_Returns_unmean_{}'.format(i), Returns_unmean, cumulative_timestep)
                    writer.add_scalar('Reward/ORCL_Returns_{}'.format(i), Returns, cumulative_timestep)

                if bandit_eval:
                    for i in range(num_tasks):
                        Returns_bandit =  episodes_return_mean_bandit[i]
                        Returns_bandit_unmean = episode_return_bandit[i]
                        c = corrects_bandit[i]
                        writer.add_scalar('Reward/BNDT_Returns_{}'.format(i), Returns_bandit, cumulative_timestep)
                        writer.add_scalar('Reward/BNDT_Returns_unmean_{}'.format(i), Returns_bandit_unmean, cumulative_timestep)
                        writer.add_scalar('Reward/BNDT_correct_arm_{}'.format(i), np.mean(c), cumulative_timestep)
"""
                # # 打印输出
                # with np.printoptions(precision=2, suppress=True):
                #     print('Task {}, Episode {}, Samples {}, Train RPE: {:.3f}, Number of Policy Updates: {}, Cumulative time steps: {}'.format(task_number,
                #           i_episode, samples_number, np.mean(list(episode_reward)), n_updates, cumulative_timestep))

def make_env(env_id):

    carla_gym_parser = argparse.ArgumentParser()

    # 注意对应每个Carla环境的-p和-tm_port !!!!!!!!
    if env_id == 1:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5000, type=int,
                                      help='TCP port to listen to (default: 2000)')
        carla_gym_parser.add_argument('--tm_port', default=16199, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 8000)')
    elif env_id == 2:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='tools_env_2/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5400, type=int,
                                      help='TCP port to listen to (default: 2000)')
        carla_gym_parser.add_argument('--tm_port', default=16399, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 8000)')
    elif env_id == 3:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='tools_env_3/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5500, type=int,
                                      help='TCP port to listen to (default: 2000)')
        carla_gym_parser.add_argument('--tm_port', default=16499, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 8000)')
    elif env_id == 4:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='tools_env_4/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5600, type=int,
                                      help='TCP port to listen to (default: 2000)')
        carla_gym_parser.add_argument('--tm_port', default=16599, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 8000)')
    else:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='tools_env_0/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5200, type=int,
                                      help='TCP port to listen to (default: 2000)')
        carla_gym_parser.add_argument('--tm_port', default=16199, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 8000)')
    carla_gym_parser.add_argument('--num_timesteps', type=float, default=1e7)
    carla_gym_parser.add_argument('--test', default=False, action='store_true')
    carla_gym_parser.add_argument('--agent_id', type=int, default=1),
    carla_gym_parser.add_argument('--carla_host', metavar='H', default='127.0.0.1',
                                  help='IP of the host server (default: 127.0.0.1)')
    carla_gym_parser.add_argument('--play_mode', type=int, help='Display mode: 0:off, 1:2D, 2:3D ', default=0)
    carla_gym_parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720',
                                  help='window resolution (default: 1280x720)')

    carla_gym_args = carla_gym_parser.parse_args()
    carla_gym_args.num_timesteps = int(carla_gym_args.num_timesteps)

    if carla_gym_args.test and carla_gym_args.cfg_file is None:
        path = 'logs/agent_{}/'.format(carla_gym_args.agent_id)
        conf_list = [cfg_file for cfg_file in os.listdir(path) if '.yaml' in cfg_file]
        carla_gym_args.cfg_file = path + conf_list[0]
    cfg_from_yaml_file(carla_gym_args.cfg_file, cfg)
    cfg.EXP_GROUP_PATH = '/'.join(carla_gym_args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if env_id == 1:
        print('Env1 is starting')
        env = gym.make("gym_env-v1")
    elif env_id == 2:
        print('Env2 is starting')
        env = gym.make('gym_env-v2')
    elif env_id == 3:
        print('Env3 is starting')
        env = gym.make('gym_env-v3')
    elif env_id == 4:
        print('Env4 is starting')
        env = gym.make('gym_env-v4')
    else:
        print('Env0 is starting')
        env = gym.make('gym_env-v0')

    env.begin_modules(carla_gym_args)

    if carla_gym_args.play_mode:
        env.enable_auto_render()

    return env

def main():
    owl_parser = argparse.ArgumentParser()
    # env params
    owl_parser.add_argument('--episode_max_steps', type=int, default=800)
    owl_parser.add_argument('--log_interval', type=int, default=200)
    owl_parser.add_argument('--n_episodes', type=int, default=1)
    owl_parser.add_argument('--n_collect_steps', type=int, default=1)
    owl_parser.add_argument('--env', dest='env', default='multi_task_car_following', type=str, help='The multi-task car following env.')
    owl_parser.add_argument('--save_model', dest='save_model', action='store_true', default=True)
    owl_parser.add_argument('--tag', type=str, default='sac_ewc_car_following_env_0', help='unique str to tag tb.')
    owl_parser.add_argument('--logdir', dest='logdir', default='runs_single_task_car_following_env_0', type=str,
                            help='The log directory for tensorboard.')
    # CL params
    owl_parser.add_argument('--num_tasks', type=int, default=1)
    owl_parser.add_argument('--num_task_repeats', type=int, default=1)
    owl_parser.add_argument('--max_task_frames', type=int, default=100000)
    owl_parser.add_argument('--q_ewc_reg', dest='q_ewc_reg', type=float, default=10000000,
                            help='EWC Q-func regularisation strength.')
    owl_parser.add_argument('--buffer_warm_start', dest='buffer_warm_start', default=False, action='store_true',
                            help='Whether to warm start the buffer when we see the same task again.')
    owl_parser.add_argument('--buffer_warm_start_size', dest='buffer_warm_start_size', type=int, default=1000,
                            help='Size of the buffer used for warm starting.')
    owl_parser.add_argument('--owl', dest='owl', default=True, action='store_true',
                            help='Whether to use owl SAC agent for CL.')
    # Bandit params
    owl_parser.add_argument('--bandits', dest='bandits', default=False, action='store_true',
                            help='Whether to use bandits to select correct head for task.')
    owl_parser.add_argument('--greedy_bandit', dest='greedy_bandit', default=False, action='store_true',
                            help='Whether to use the greedy bandit.')
    owl_parser.add_argument('--bandit_loss', dest='bandit_loss', type=str, default='mse',
                            help='Bandit loss for Exp weights \in {nll, mse, hack}.')
    owl_parser.add_argument('--bandit_lr', dest='bandit_lr', type=float, default=0.88, help='Bandit learning rate.')
    owl_parser.add_argument('--bandit_decay', dest='bandit_decay', type=float, default=0.9, help='Bandit decay.')
    owl_parser.add_argument('--bandit_step', dest='bandit_step', type=int, default=1,
                            help='Number of steps taken between bandit arm pulls.')
    owl_parser.add_argument('--bandit_epsilon', dest='bandit_epsilon', type=float, default=0.0,
                            help='Eps greedy exploration in the ExpWeights bandit alg.')
    owl_args = owl_parser.parse_args()

    owl_params = vars(owl_args)  # 将args转换为字典类型，params是args的字典形式

    envs = []

    if owl_args.env == 'multi_task_car_following':
        env_ids = [1]
        for env_id in env_ids:
            env = make_env(env_id)
            state_dim = env.observation_space.shape[1]
            action_dim = env.action_space.shape[1]  # 在对抗场景的环境中该数值为2,动态修改
            max_action = env.action_space.high[0]
            min_action = env.action_space.low[0]
            max_action = torch.Tensor(max_action)
            min_action = torch.Tensor(min_action)
            # state_dim = 3
            # action_dim = 1
            # max_action = 3
            # min_action = -3
            print("Env{}: state_dim: {}, action_dim: {}，max_action: {}, min_action: {}.".format(env_id, state_dim, action_dim, max_action, min_action))
            envs.append(env)

    assert len(envs) == owl_args.num_tasks

    if owl_args.owl:
        agent = SACAgentOWL(state_dim, action_dim, max_action, min_action, num_tasks=owl_args.num_tasks,
                            hidden_size=256, actor_lr=0.0003, critic_lr=0.0003, q_ewc_reg=owl_params['q_ewc_reg'], n_fisher_sample=10000,
                            gamma=0.99, q_lr=3e-4, batch_size=1, tau=0.005, adaptive_alpha=True, pool_size=100000)
    else:
        agent = SACAgent(state_dim=3, action_dim=1, max_action=3.0, min_action=-3.0, num_tasks=1, hidden_size=256, actor_lr=0.0003,
                         critic_lr=0.0003, q_ewc_reg=owl_params['q_ewc_reg'], gamma=0.99, q_lr=3e-4, batch_size=256, tau=0.005, adaptive_alpha=True, pool_size=100000)

    train_agent_model_free_multi_task_car_following(agent=agent, envs=envs, params=owl_params)

if __name__ == '__main__':
    main()
