import pickle
import numpy as np
from stable_baselines3.common.save_util import save_to_pkl





path1 = './Training_Results/save_replay_buffer/single_scenario/rl_model_replay_buffer_2300000_steps.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
f1 = open(path1, 'rb')
data = pickle.load(f1)


path2 = './Training_Results/save_replay_buffer/single_scenario/rl_model_replay_buffer_1800000_steps.pkl'
f2 = open(path2, 'rb')
data2 = pickle.load(f2)


data.actions = np.concatenate((data.actions[:data.pos,:,:],data2.actions[:-data.pos,:,:]))
data.observations = np.concatenate((data.observations[:data.pos,:,:,:], data2.observations[1:-data.pos+1,:,:,:]))
data.next_observations = np.concatenate((data.next_observations[:data.pos,:,:,:],data2.next_observations[1:-data.pos+1,:,:,:]))
data.rewards = np.concatenate((data.rewards[:data.pos,:],data2.rewards[:-data.pos,:]))
data.pos = min(data.buffer_size-1, (data.pos + data2.pos - 1))
data.full = True


path = './Training_Results/save_replay_buffer/save_replay_buffer_80_20.pkl'
save_to_pkl(path, data, 0)

path0 = './Training_Results/save_replay_buffer/save_replay_buffer_80_20.pkl'
f = open(path0, 'rb')
data0 = pickle.load(f)

print(data)
# print(len(data))
