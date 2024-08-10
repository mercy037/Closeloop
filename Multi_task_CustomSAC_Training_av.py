'''在对抗或自然环境下训练自动驾驶汽车'''
import time
import gym
import argparse
import os
import numpy as np
import pandas as pd
from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
# from stable_baselines3 import deepq
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from SwitchEnvforCL import SwitchEnvCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

import Carla_gym
from matplotlib import pyplot as plt
from stable_baselines3 import SAC
from EWC_CustomSAC import CustomSAC
from EWC_CustomSAC import CustomSACPolicy
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args_cfgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                        help='specify the config for training')
    parser.add_argument('--num_timesteps', type=float, default=1e7)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--agent_id', type=int, default=1),
    parser.add_argument('-p', '--carla_port', metavar='P', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--tm_port', default=8000, type=int,
                        help='Traffic Manager TCP port to listen to (default: 8000)')
    parser.add_argument('--carla_host', metavar='H', default='127.0.0.1',
                        help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('--play_mode', type=int, help='Display mode: 0:off, 1:2D, 2:3D ', default=0)
    parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720',
                        help='window resolution (default: 1280x720)')

    args = parser.parse_args()
    args.num_timesteps = int(args.num_timesteps)

    if args.test and args.cfg_file is None:
        path = 'logs/agent_{}/'.format(args.agent_id)
        conf_list = [cfg_file for cfg_file in os.listdir(path) if '.yaml' in cfg_file]
        args.cfg_file = path + conf_list[0]
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    return args, cfg


def make_env(env_id):
    carla_gym_parser = argparse.ArgumentParser()

    # 注意对应每个Carla环境的-p和-tm_port !!!!!!!!
    if env_id == 1:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5100, type=int,
                                      help='TCP port to listen to (default: 5100)')
        carla_gym_parser.add_argument('--tm_port', default=16100, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 16100)')
    elif env_id == 2:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5200, type=int,
                                      help='TCP port to listen to (default: 5200)')
        carla_gym_parser.add_argument('--tm_port', default=16200, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 16200)')
    elif env_id == 3:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5300, type=int,
                                      help='TCP port to listen to (default: 5300)')
        carla_gym_parser.add_argument('--tm_port', default=16300, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 16300)')
    elif env_id == 4:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5400, type=int,
                                      help='TCP port to listen to (default: 5400)')
        carla_gym_parser.add_argument('--tm_port', default=16400, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 16400)')

    elif env_id == 5:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5500, type=int,
                                      help='TCP port to listen to (default: 5400)')
        carla_gym_parser.add_argument('--tm_port', default=16500, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 16400)')

    elif env_id == 7:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5700, type=int,
                                      help='TCP port to listen to (default: 5400)')
        carla_gym_parser.add_argument('--tm_port', default=16700, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 16400)')

    elif env_id == 8:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5800, type=int,
                                      help='TCP port to listen to (default: 5400)')
        carla_gym_parser.add_argument('--tm_port', default=16800, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 16400)')

    elif env_id == 10:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5110, type=int,
                                      help='TCP port to listen to (default: 5400)')
        carla_gym_parser.add_argument('--tm_port', default=16110, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 16400)')

    else:
        carla_gym_parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/config.yaml',
                                      help='specify the config for training')
        carla_gym_parser.add_argument('-p', '--carla_port', metavar='P', default=5000, type=int,
                                      help='TCP port to listen to (default: 5000)')
        carla_gym_parser.add_argument('--tm_port', default=16000, type=int,
                                      help='Traffic Manager TCP port to listen to (default: 16000)')
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
        env = gym.make('gym_env-v1')
    elif env_id == 2:
        print('Env2 is starting')
        env = gym.make('gym_env-v2')
    elif env_id == 3:
        print('Env3 is starting')
        env = gym.make('gym_env-v3')
    elif env_id == 4:
        print('Env4 is starting')
        env = gym.make('gym_env-v4')
    elif env_id == 5:
        print('Env5 is starting')
        env = gym.make('gym_env-v5')
    elif env_id == 7:
        print('Env7 is starting')
        env = gym.make('gym_env-v7')
    elif env_id == 8:
        print('Env8 is starting')
        env = gym.make('gym_env-v8')
    elif env_id == 10:
        print('Env8 is starting')
        env = gym.make('gym_env-v10')
    else:
        print('Env0 is starting')
        env = gym.make('gym_env-v0')

    env.begin_modules(carla_gym_args)

    if carla_gym_args.play_mode:
        env.enable_auto_render()

    return env


if __name__ == '__main__':
    # args, cfg = parse_args_cfgs()
    owl_parser = argparse.ArgumentParser()
    # CL params
    owl_parser.add_argument('--num_tasks', type=int, default=2)
    owl_parser.add_argument('--num_task_repeats', type=int, default=2)
    owl_parser.add_argument('--max_task_frames', type=int, default=1000000)
    owl_parser.add_argument('--q_ewc_reg', dest='q_ewc_reg', type=float, default=0.001,
                            help='EWC Q-func regularisation strength.')
    owl_parser.add_argument('--buffer_warm_start', dest='buffer_warm_start', default=False, action='store_true',
                            help='Whether to warm start the buffer when we see the same task again.')
    owl_parser.add_argument('--buffer_warm_start_size', dest='buffer_warm_start_size', type=int, default=1000,
                            help='Size of the buffer used for warm starting.')
    owl_parser.add_argument('--owl', dest='owl', default=True, action='store_true',
                            help='Whether to use owl SAC agent for CL.')
    owl_args = owl_parser.parse_args()
    owl_params = vars(owl_args)  # 将args转换为字典类型，params是args的字典形式

    print('Env is starting')
    env_multi = []
    env_ids = [1, 10, 4]
    for env_id in env_ids:
        env = make_env(env_id)
        print("Env{} is made.".format(env_id))
        env_multi.append(env)
    # create a VecEnv object with SubprocVecEnv
    # env_multi_vector = [lambda: env_multi[0], lambda: env_multi[1]] #DummyVecEnv需要传入函数参数
    # venv = DummyVecEnv(env_multi_vector)
    num_tasks = len(env_ids)

    train_npc = 1   # 训练NPC的标志,0测试，1训练，2继续训练
    actor_q_ewc_reg = 5
    critic_q_ewc_reg = 0
    policy_kwargs = dict(num_tasks=num_tasks, actor_q_ewc_reg=actor_q_ewc_reg, critic_q_ewc_reg=critic_q_ewc_reg)
    # 1.生成一个新的模型并进行训练
    if train_npc == 1:
        model = CustomSAC("CustomSACPolicy", env=env_multi[0], policy_kwargs=policy_kwargs,
                          num_tasks=num_tasks, actor_q_ewc_reg=actor_q_ewc_reg, critic_q_ewc_reg=critic_q_ewc_reg,
                          n_fisher_sample=300000,
                          tensorboard_log="./Training_Results/final/Tensorboard_Results/", verbose=1)
        checkpoint_callback = CheckpointCallback(save_freq=500000,
                                                 save_path='./Training_Results/final/AV_Model_SAC',
                                                 verbose=1)
        # create a SwitchEnvCallback object with the list of environments and the switch frequency
        switch_env_callback = SwitchEnvCallback(env_multi, switch_freq=1500000)
        # create a CallbackList object with the list of callbacks
        callback_list = CallbackList([checkpoint_callback, switch_env_callback])
        model.learn(total_timesteps=10000000, log_interval=1, callback=callback_list,
                    tb_log_name="self-evolution", reset_num_timesteps=False)
        model.save("./Training_Results/final/AV_Model_SAC_final")

    # 2.加载已有模型，在其基础上进行训练
    if train_npc == 2:
        model = CustomSAC.load("./Training_Results/0317/AV_Model_SAC/rl_model_1500000_steps.zip", env=env_multi[0],
                               device=device)
        model.num_timesteps = 0
        q_ewc_reg = 0.0001      # 0.0001 15   0.001 16  0.01 17 0.1 18 失败 0.05 19
        model.q_ewc_reg = q_ewc_reg
        model.actor.q_ewc_reg = q_ewc_reg
        model.critic.q_ewc_reg = q_ewc_reg
        model.critic_target.q_ewc_reg = q_ewc_reg
        num_tasks = len(env_ids)
        model.num_tasks = num_tasks
        model.actor.num_tasks = num_tasks
        model.critic.num_tasks = num_tasks
        model.critic_target.num_tasks = num_tasks
        checkpoint_callback = CheckpointCallback(save_freq=250000,
                                                 save_path='/data/csy/Training_Results/0516/AV_Model_SAC', verbose=1)
        switch_env_callback = SwitchEnvCallback(env_multi, switch_freq=500000)
        switch_env_callback.num_steps = model.num_timesteps
        callback_list = CallbackList([checkpoint_callback, switch_env_callback])
        model.learn(total_timesteps=9000000, log_interval=1, callback=callback_list, tb_log_name="train_0516", reset_num_timesteps=False)
        model.save("./Training_Results/0516/AV_Model_SAC_final")
    # 0. 加载已有模型，进行测试
    elif train_npc == 0:
        total_reward = 0
        step = 0
        model = CustomSAC.load("./Training_Results/0521_fisher_2/AV_Model_SAC/rl_model_1500000_steps.zip", env=env_multi[0])
        env = env_multi[0]
        obs = env.reset()  # 初始化环境
        data = []
        episode = 0
        try:
            while True:
                obs = np.array(obs).reshape(1, 30)
                action, _ = model.predict(obs)
                obs, reward, done, result_info = env.step(action)
                # print("Observation:",obs)
                total_reward += reward
                info = result_info["info"]
                if done:
                    # print("Episode结束，环境重置")
                    env.reset()
                    total_reward = 0
                    step = 0
                    if info["Collision"] == True and cfg.OTHER_INFOMATION.RECORD == 1:
                        episode += 1
                        Data = pd.DataFrame(list(data))
                        filename = './' + str(episode) + '.xls'
                        Data.to_excel(filename, header=False, index=False)
                        print("Collision Scenario Successfully Recorded")
                    data = []
                step += 1
                data.append(info["DataInfo"])

                # print("Observation:",obs)
        finally:
            print('测试结束，存储数据')
