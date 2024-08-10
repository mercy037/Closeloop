'''在对抗或自然环境下训练自动驾驶汽车'''
import time
import gym
import argparse
import os
from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from stable_baselines3 import PPO
from stable_baselines3 import SAC

# from stable_baselines3 import deepq
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd

import Carla_gym
from matplotlib import pyplot as plt

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


if __name__ == '__main__':
    train_npc = 4 # 训练NPC的标志,0测试，1训练，2继续训练
    args, cfg = parse_args_cfgs()
    print('Env is starting')
    env = gym.make("gym_env-v1")
    if args.play_mode:
        env.enable_auto_render()
    env.begin_modules(args)

    # 4.尝试修改官方train过程
    if train_npc == 4:
        model = SAC("MlpPolicy", env=env,  tensorboard_log="./Training_Results/0718-2/Tensorboard_Results/", verbose = 1)
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./Training_Results/0718-2/AV_Model_SAC' ,verbose = 1)
        model.learn(total_timesteps=2000000, log_interval=1, callback = checkpoint_callback, reset_num_timesteps = False)
        model.save("./Training_Results/0718-2/AV_Model_SAC_final")

    # 1.生成一个新的模型并进行训练
    if train_npc == 1:
        model = SAC("MlpPolicy", env=env,  tensorboard_log="./Training_Results/0718-2/Tensorboard_Results/", verbose = 1)
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./Training_Results/0718-2/AV_Model_SAC' ,verbose = 1)
        model.learn(total_timesteps=2000000, log_interval=1, callback = checkpoint_callback, reset_num_timesteps = False)
        model.save("./Training_Results/0718-2/AV_Model_SAC_final")

    # 2.加载已有模型，在其基础上进行训练
    if train_npc == 2:
        model = SAC.load("./Training_Results/0718-2/AV_Model_SAC/rl_model_1550000_steps.zip" , env=env, device = device)
        checkpoint_callback = CheckpointCallback(save_freq=20000, save_path='./Training_Results/0718-2/AV_Model_SAC', verbose = 1)
        model.learn(total_timesteps=1000000, log_interval=1, callback=checkpoint_callback, reset_num_timesteps=False)
        model.save("./Training_Results/0718-2/AV_Model_SAC_final")

   # 0. 加载已有模型，进行测试
    elif train_npc == 0:
        total_reward = 0
        step = 0
        model = SAC.load("./Training_Results/0712-3/AV_Model_SAC/rl_model_1500000_steps.zip", env=env)
        obs = env.reset() #初始化环境
        data = []
        episode = 0
        try:
            while True:
                obs = np.array(obs).reshape(1,30)
                action,_ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                # print("Observation:",obs)
                total_reward += reward
                if done:
                    # print("Episode结束，环境重置")
                    env.reset()
                    total_reward = 0
                    step = 0
                    if info["Collision"] == True and cfg.OTHER_INFOMATION.RECORD == 1:
                        episode += 1
                        Data = pd.DataFrame(list(data))
                        filename = './'+ str(episode) +'.xls'
                        Data.to_excel(filename, header=False, index=False)
                        print("Collision Scenario Successfully Recorded")
                    data = []
                step += 1
                data.append(info["DataInfo"])

                # print("Observation:",obs)
        finally:
            print('测试结束，存储数据')


