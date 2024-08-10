'''以自动驾驶汽车为环境训练对抗周车代理'''
import time
import gym
import argparse
import os
from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import pandas as pd

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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:2')

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
    train_npc = 1 # 训练NPC的标志,0测试，1训练，2继续训练
    args, cfg = parse_args_cfgs()
    print('Env is starting')
    env = gym.make("gym_env-v5")
    if args.play_mode:
        env.enable_auto_render()
    env.begin_modules(args)

    # 1.生成一个新的模型并进行训练
    if train_npc == 1:
        model = SAC("MlpPolicy", env=env,  tensorboard_log="./Training_Results/0119/Tensorboard_Results/", verbose = 1, device = device)
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./Training_Results/0119/AV_Model_SAC' ,verbose = 1)
        model.learn(total_timesteps=1000000, log_interval=1, callback = checkpoint_callback, reset_num_timesteps = False)
        model.save("./Training_Results/0119/BV_Model_SAC_final")

    # 2.加载已有模型，在其基础上进行训练
    if train_npc == 2:
        model = SAC.load("./Training_Results/0716-1/AV_Model_SAC/rl_model_900000_steps.zip" , env=env)
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./Training_Results/0712-3/AV_Model_SAC', verbose = 1, device = device)
        model.learn(total_timesteps=1000000, log_interval=1, callback=checkpoint_callback, reset_num_timesteps=False)
        model.save("./Training_Results/0712-3/AV_Model_SAC_final")

   # 0. 加载已有模型，进行测试
    elif train_npc == 0:
        total_reward = 0
        step = 0
        model = SAC.load("./Training_Results/0716-1/AV_Model_SAC/rl_model_910000_steps.zip", env=env,device = device)
        obs = env.reset() #初始化环境
        data = []
        data_idx = 0
        try:
            while True:
                obs = np.array(obs).reshape(1,30)
                action,_ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                # print("Observation:",obs)
                total_reward += reward
                step += 1
                data.append(info["DataInfo"])
                if done:
                    # print("Episode结束，环境重置")
                    env.reset()
                    total_reward = 0
                    step = 0
                    if info["Collision"] == True and cfg.OTHER_INFOMATION.RECORD == 1:
                        data_idx  += 1
                        Data = pd.DataFrame(list(data))
                        filename = './Scenario_Analyse/Raw_Scenario_Data/Autonomous_Vehicle1/Critical_Scenario/Method1/'+ str(data_idx ) +'.csv'
                        Data.to_csv(filename, header=False, index=False)
                        print("Collision Scenario Successfully Recorded")
                    data = []
        finally:
            print('测试结束，存储数据')


