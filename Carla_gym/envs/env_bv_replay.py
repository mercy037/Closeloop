import math
import numpy as np
import time
import matplotlib.pyplot as plt
import gym
import pandas as pd
from gym import spaces
from tools.modules_1 import *
from agents.local_planner.frenet_optimal_trajectory_zg import FrenetPlanner as MotionPlanner
from agents.low_level_controller.controller import VehiclePIDController
from agents.low_level_controller.controller import PIDLongitudinalController
from agents.low_level_controller.controller import PIDLateralController
from agents.tools.misc import get_speed
from agents.low_level_controller.controller import IntelligentDriverModel
from config import cfg
from common.utils import lamp, closest, closest_wp_idx
#from MPC.MPC_controller_lon import MPC_controller_lon
#from MPC.parameter_config import MPC_lon_Config


from datas.data_log import data_collection
from stable_baselines3 import SAC
#MPC_lon_Config = MPC_lon_Config()
from agents.local_planner.frenet_optimal_trajectory_lon import velocity_inertial_to_frenet, \
    get_obj_S_yaw
import heapq
import torch
MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'
TENSOR_ROW_NAMES = ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN',
                    'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN']

def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb

def cal_lat_error(waypoint1, waypoint2, vehicle_transform):
    """
    Estimate the steering angle of the vehicle based on the PID equations

    :param waypoint: target waypoint [x, y]
    :param vehicle_transform: current transform of the vehicle
    :return: lat_error
    """
    v_begin = vehicle_transform.location
    v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                     y=math.sin(math.radians(vehicle_transform.rotation.yaw)))
    v_vec_0 = np.array([math.cos(math.radians(vehicle_transform.rotation.yaw)), math.sin(math.radians(vehicle_transform.rotation.yaw)), 0.0])
    v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
    w_vec = np.array([waypoint2[0] -
                      waypoint1[0], waypoint2[1] -
                      waypoint1[1], 0.0])
    lat_error = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                  (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

    return lat_error

def update_adversarial_bv(actors_batch, x_av, y_av,idx):
    update = False # 是否更新对手的标志
    #首先判断是否要更新idx：更新条件:原本对抗车辆的位置不满足要求，或idx = -1即没有选择出对抗车辆
    if idx >= 0 :
        # print(idx)
        x_bv, y_bv = float(actors_batch[idx]['Obj_Cartesian_state'][0]), float(actors_batch[idx]['Obj_Cartesian_state'][1])
        # 如果当前对抗车辆的位置位于指定区域内，不更新对抗车辆序号
        if  -2.5 < (x_bv - x_av) < 15 and -5.25 < (y_bv - y_av) < 5.25 and 24.5 < y_bv < 39:
            update = False
        # 如果当前车辆的位置抽出了指定范围内，更新对抗车辆序号
        else:
            update = True
    else:
        update = True
    # update = True # 在训练结束后，也可以直接将update标志设置为true，在所有时刻切换车辆
    # 下面为对抗车辆的选择过程：选择出指定区域内距离自车最近的车辆作为idx
    distance_bv = []
    # 如果更新标志为真，重新选择一辆对手
    if update == True:
        for i, actor in enumerate(actors_batch):
            x_bv, y_bv = float(actor['Obj_Cartesian_state'][0]), float(actor['Obj_Cartesian_state'][1])
            if  -2.5 < (x_bv - x_av) < 15 and -5.25 < (y_bv - y_av) < 5.25 and 24.5 < y_bv < 39:
                distance = math.sqrt((x_bv - x_av) ** 2 + (y_bv - y_av) ** 2)
            else:
                distance = 10000  # 计算bv和av的质心距离
            distance_bv.append(distance)  # 存储了每个车辆距离自车的距离
        if min(distance_bv) == 10000: # 如果bv和av的最近距离为10000，说明没有车辆满足要求，idx改为-1，不存在对抗车辆
            idx_new = -1
        else:
            idx_new = distance_bv.index(min(distance_bv))  # 选出距离自车最近的车辆

    # 如果更新标志为假，将上一个时刻的对抗车辆作为下一个时刻的对抗车辆
    else:
            idx_new = idx

    return idx_new

def select_nearest_bv(actors_batch,x_av,y_av,idx_bv): # 选出自车前后五十米内距离自车最近的四辆车
    distance_bv = []
    for i, actor in enumerate(actors_batch):
        x_bv, y_bv = float(actor['Obj_Cartesian_state'][0]), float(actor['Obj_Cartesian_state'][1])
        distance = math.sqrt((x_bv - x_av) ** 2 + (y_bv - y_av) ** 2)  # 计算bv和av的质心距离
        if (x_bv-x_av) > 50 or (x_bv-x_av) <-50:
            distance = 10000
        if i == idx_bv :
            distance = 20000
        distance_bv.append(distance)  # 存储了每个车辆距离自车的距离
    num_bv = cfg.TRAFFIC_MANAGER.N_SPAWN_CARS
    if distance_bv == []:
        distance_bv = [10000]

    if num_bv < 4:
        if num_bv == 1:
            min1= heapq.nsmallest(num_bv, distance_bv)[0]
            min2, min3, min4 = None,None,None
        if num_bv == 2:
            min1, min2 = heapq.nsmallest(num_bv, distance_bv)
            min3, min4 = None, None
        if num_bv == 3:
            min1, min2, min3 = heapq.nsmallest(num_bv, distance_bv)
            min4 = None

    else:
        min1, min2, min3, min4 = heapq.nsmallest(4,distance_bv)

    if min1 == 10000 or min1 == None:
        idx1 = -1
    else:
        idx1 = distance_bv.index(min1)
    if min2 == 10000 or min2 == None:
        idx2 = -1
    else:
        idx2 = distance_bv.index(min2)
    if min3 == 10000 or min3 == None:
        idx3 = -1
    else:
        idx3 = distance_bv.index(min3)
    if min4 == 10000 or min4 == None:
        idx4 = -1
    else:
        idx4 = distance_bv.index(min4)


    return idx1, idx2, idx3, idx4

class CarlagymEnv3(gym.Env):

    # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.idx_bv = -1
        self.du_lon_last = None
        self.Input_lon = None
        self.lat_error = None
        self.__version__ = "9.9.2"

        # simulation
        self.verbosity = 0
        self.auto_render = False  # automatically render the environment
        self.n_step = 0
        try:
            self.global_route = np.load(
                'road_maps/global_route_town04.npy')  # track waypoints (center lane of the second lane from left)
            # 1520 *  3

        except IOError:
            self.global_route = None

        # constraints

        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.minSpeed = float(cfg.GYM_ENV.MIN_SPEED)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)

        self.INIT_ACTION = cfg.TRAFFIC_MANAGER.INIT_ACTION

        # frenet
        self.f_idx = 0
        self.init_s = None  # initial frenet s value - will be updated in reset function
        self.max_s = int(cfg.CARLA.MAX_S)#self.max_s指3000米，最大的s值
        self.effective_distance_from_vehicle_ahead = int(cfg.GYM_ENV.DISTN_FRM_VHCL_AHD)
        self.d_max_s = int(cfg.CARLA.D_MAX_S)
        self.lanechange = False
        self.is_first_path = True
        self.obj_max_vs = int(cfg.TRAFFIC_MANAGER.MAX_SPEED_2)
        self.last_num_overtake = 0
        self.last_df_n = 0.0

        # RL
        self.collision_penalty = int(cfg.RL.COLLISION)
        self.low_speed_reward = float(cfg.RL.Low_SPEED_REWARD)
        self.middle_speed_reward = float(cfg.RL.Middle_SPEED_REWARD)
        self.high_speed_reward = float(cfg.RL.High_SPEED_REWARD)
        self.total_reward = 0
        # self.lane_change_reward = float(cfg.RL.LANE_CHANGE_REWARD)
        # self.lane_change_penalty = float(cfg.RL.LANE_CHANGE_PENALTY)
        # self.off_the_road_penalty = int(cfg.RL.OFF_THE_ROAD)

        # instances
        self.ego = None
        self.actors_batch = None
        self.ego_los_sensor = None
        self.module_manager = None
        self.world_module = None
        self.traffic_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None  # ego initial transform to recover at each episode
        self.acceleration_ = 0
        self.eps_rew = 0
        self.u_lon_last = 0.0
        self.u_lon_llast = 0.0
        self.fig, self.ax = plt.subplots()
        self.x = []
        self.y = []
        self.motionPlanner = None
        self.vehicleController = None
        self.PIDLongitudinalController = None
        self.PIDLateralController = None
        self.num_collision = 0
        self.num_episode = 0

        # 读取文件中的数据
        data = pd.read_excel(cfg.OTHER_INFOMATION.DATA_NAME)  # 默认读取第一个sheet
        self.data = np.array(data)
        self.init_t = int(cfg.OTHER_INFOMATION.INIT_TIME_STEP)


        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05

        """连续动作空间"""
        action_low = -1
        action_high = 1

        self.action_dim = (1, 2)
        self.action_space = spaces.Box(action_low, action_high, shape=self.action_dim,
                                       dtype='float32')  # [long and lat]  [-1, 1]

        self.obs_dim = (1, 30)  # [ego_s, ego_d, v_s, v_d, phi_frenet, obj_s, obj_d, obj_v_s, obj_v_d, obj_phi_frenet]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.obs_dim, dtype='float32')
        self.state = np.zeros_like(self.observation_space.sample())

    def obs_nearest_five_bv(self):
        '''计算AV的观察'''
        distance_bv = []

        '''获取AV的状态'''
        x_av = self.ego.get_location().x
        y_av = self.ego.get_location().y
        vx_av = self.ego.get_velocity().x
        vy_av = self.ego.get_velocity().y
        psi = math.radians(self.ego.get_transform().rotation.yaw)

        '''找出距离AV最近的五辆车'''
        actors_batch = self.traffic_module.actors_batch
        # print(len(actors_batch))
        for i, actor in enumerate(actors_batch):
            x_bv, y_bv = float(actor['Obj_Cartesian_state'][0]), float(actor['Obj_Cartesian_state'][1])
            distance = math.sqrt((x_bv - x_av) ** 2 + (y_bv - y_av) ** 2)  # 计算bv和av的质心距离
            distance_bv.append(distance)  # 存储了每个车辆距离自车的距离
        num_bv = cfg.TRAFFIC_MANAGER.N_SPAWN_CARS
        assert num_bv >= 5, "总车数少于五"
        min1, min2, min3, min4, min5 = heapq.nsmallest(5, distance_bv)  # 找出出距离自车最近的五辆周车的编号
        idx1 = distance_bv.index(min1)
        idx2 = distance_bv.index(min2)
        idx3 = distance_bv.index(min3)
        idx4 = distance_bv.index(min4)
        idx5 = distance_bv.index(min5)

        '''Observation for AV'''
        av_info = [x_av,y_av,vx_av,vy_av,psi]

        others_x = np.zeros(self.N_SPAWN_CARS)
        others_y = np.zeros(self.N_SPAWN_CARS)
        others_vx = np.zeros(self.N_SPAWN_CARS)
        others_vy = np.zeros(self.N_SPAWN_CARS)
        others_psi = np.zeros(self.N_SPAWN_CARS)

        for i, actor in enumerate(self.traffic_module.actors_batch):
            x_bv, y_bv, _, vx_bv, vy_bv, _, _, _, psi_bv, _, _, _ = actor["Obj_Cartesian_state"]

            others_x[i] = x_bv
            others_y[i] = y_bv
            others_vx[i] = vx_bv
            others_vy[i] = vy_bv
            others_psi[i] = psi_bv

        # 计算观察,包括自车状态以及周车相对于自车的相对状态


        bv1_info = [others_x[idx1]-x_av, others_y[idx1]-y_av, others_vx[idx1]-vx_av,others_vy[idx1]-vy_av,others_psi[idx1]-psi]
        bv2_info = [others_x[idx2]-x_av, others_y[idx2]-y_av, others_vx[idx2]-vx_av,others_vy[idx2]-vy_av,others_psi[idx2]-psi]
        bv3_info = [others_x[idx3]-x_av, others_y[idx3]-y_av, others_vx[idx3]-vx_av,others_vy[idx3]-vy_av,others_psi[idx3]-psi]
        bv4_info = [others_x[idx4]-x_av, others_y[idx4]-y_av, others_vx[idx4]-vx_av,others_vy[idx4]-vy_av,others_psi[idx4]-psi]
        bv5_info = [others_x[idx5]-x_av, others_y[idx5]-y_av, others_vx[idx5]-vx_av,others_vy[idx5]-vy_av,others_psi[idx5]-psi]


        obs_av = av_info + bv1_info + bv2_info + bv3_info + bv4_info + bv5_info

        return obs_av

    def AV_action_to_control(self,action_av):
        '''Initial Flags'''
        self.f_idx = 1
        loop_counter = 0
        action_av = np.array(action_av)


        ''' Action Design '''

        if action_av.ndim == 2:
            df_n = action_av[0][0] * 3.5
            acc_input = np.clip(action_av[0][1] * 8.0, -8.0, 2.0)
        else:
            df_n = action_av[0] * 3.5
            acc_input = np.clip(action_av[1] * 8.0, -8.0, 2.0)

        ''' Planner '''
        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        angular_velocity = self.ego.get_angular_velocity()
        acc_angular = math.sqrt(angular_velocity.x ** 2 + angular_velocity.y ** 2 + angular_velocity.z ** 2)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, self.max_s]
        # 计算期望的侧向移动量
        fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path(ego_state, self.f_idx,
                                                                                       df_n,
                                                                                       Tf=5,
                                                                                       Vf_n=0)  # 根据当前车辆状态进行路径规划，自车的状态
        '''Observation 1 for AV'''
        # 第一种观察空间，维度为27
        vx_ego = self.ego.get_velocity().x
        vy_ego = self.ego.get_velocity().y
        vz_ego = self.ego.get_velocity().z
        ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]  # estimated current ego_s
        ego_d = fpath.d[self.f_idx]
        v_S, v_D = velocity_inertial_to_frenet(ego_s, vx_ego, vy_ego, self.motionPlanner.csp)
        state_vector = self.state_input_vector(v_S, ego_s, ego_d)

        for i in range(len(state_vector)):
            self.state[0][i] = state_vector[i]
        self.obs_av = self.state

        '''Controller'''
        while loop_counter < 1:
            loop_counter += 1
            ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                         math.radians(self.ego.get_transform().rotation.yaw), 0, 0, temp, self.max_s]

            self.f_idx = closest_wp_idx(ego_state, fpath, self.f_idx)
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

            '''******   overwrite command speed    ******'''
            cmdSpeed = get_speed(self.ego) + float(acc_input) * self.dt
            # control_av = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control()
            control_av = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)

        return control_av, df_n, acc_input, ego_d

    def obj_info(self):
        """
        Frenet:  [s,d,v_s, v_d, phi_Frenet]
        """
        others_s = np.zeros(self.N_SPAWN_CARS)
        others_d = np.zeros(self.N_SPAWN_CARS)
        others_v_S = np.zeros(self.N_SPAWN_CARS)
        others_v_D = np.zeros(self.N_SPAWN_CARS)
        others_phi_Frenet = np.zeros(self.N_SPAWN_CARS)

        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d, act_v_S, act_v_D, act_psi_Frenet , _ = actor['Obj_Frenet_state']
            others_s[i] = act_s
            others_d[i] = act_d
            others_v_S[i] = act_v_S
            others_v_D[i] = act_v_D
            others_phi_Frenet[i] = act_psi_Frenet
        obj_info_Mux = np.vstack((others_s, others_d, others_v_S, others_v_D, others_phi_Frenet))
        return obj_info_Mux

    def info(self, ego_s, ego_d):
        # 获取并存储自车和周车的状态信息(三个方向位置和角度坐标，三个方向的速度)，用于记录场景数据

        others_x = np.zeros(self.N_SPAWN_CARS)
        others_y = np.zeros(self.N_SPAWN_CARS)
        others_z = np.zeros(self.N_SPAWN_CARS)
        others_vx = np.zeros(self.N_SPAWN_CARS)
        others_vy = np.zeros(self.N_SPAWN_CARS)
        others_vz = np.zeros(self.N_SPAWN_CARS)
        others_pitch = np.zeros(self.N_SPAWN_CARS)
        others_roll = np.zeros(self.N_SPAWN_CARS)
        others_yaw = np.zeros(self.N_SPAWN_CARS)
        others_info_ego_s = np.zeros(self.N_SPAWN_CARS)
        others_info_ego_d = np.zeros(self.N_SPAWN_CARS)

        for i, actor in enumerate(self.traffic_module.actors_batch):

            act_x, act_y, act_z, act_v_x, act_v_y, act_v_z, act_pitch, act_roll , act_yaw, _ , _, _= actor["Obj_Cartesian_state"]

            others_x[i] = act_x
            others_y[i] = act_y
            others_z[i] = act_z
            others_vx[i] = act_v_x
            others_vy[i] = act_v_y
            others_vz[i] = act_v_z
            others_pitch[i] = act_pitch
            others_roll[i] = act_roll
            others_yaw[i] = act_yaw

        #收集了其他车辆信息后，将该向量与自车状态组合作为记录的信息返回

        ego_x = [self.ego.get_location().x]
        ego_y = [self.ego.get_location().y]
        ego_z = [self.ego.get_location().z]
        ego_vx = [self.ego.get_velocity().x]
        ego_vy = [self.ego.get_velocity().y]
        ego_vz = [self.ego.get_velocity().z]
        ego_pitch = [math.radians(self.ego.get_transform().rotation.pitch)]
        ego_roll = [math.radians(self.ego.get_transform().rotation.roll)]
        ego_yaw = [math.radians(self.ego.get_transform().rotation.yaw)]

        info_x = np.append(ego_x, others_x)
        info_y = np.append(ego_y, others_y)
        info_z = np.append(ego_z, others_z)
        info_vx = np.append(ego_vx, others_vx)
        info_vy = np.append(ego_vy,others_vy)
        info_vz = np.append(ego_vz, others_vz)
        info_pitch = np.append(ego_pitch, others_pitch)
        info_roll = np.append(ego_roll, others_roll)
        info_yaw = np.append(ego_yaw,others_yaw)

        info_ego_s = np.append(ego_s,others_info_ego_s)
        info_ego_d = np.append(ego_d, others_info_ego_d)

        obj_info_Mux = np.vstack((info_x, info_y, info_z, info_vx, info_vy, info_vz, info_pitch, info_roll, info_yaw, info_ego_s, info_ego_d))

        return obj_info_Mux

    def state_input_vector(self, v_S, ego_s, ego_d):
        # Paper: Automated Speed and Lane Change Decision Making using Deep Reinforcement Learning
        state_vector = np.zeros(27)
        state_vector[0] = v_S / self.maxSpeed

        self.df_ego = closest([self.LANE_WIDTH * lane_n for lane_n in range(-1, 3)], ego_d)

        if self.df_ego == -3.5:
            lane_num = 'Lane_1'
        elif self.df_ego == 0:
            lane_num = 'Lane_2'
        elif self.df_ego == 3.5:
            lane_num = 'Lane_3'
        elif self.df_ego == 7:
            lane_num = 'Lane_4'

        Nearby_lane_info = {'Lane_1': [0, 1], 'Lane_2': [1, 1], 'Lane_3': [1, 1], 'Lane_4': [1, 0]}
        state_vector[1] = Nearby_lane_info[lane_num][0]
        state_vector[2] = Nearby_lane_info[lane_num][1]

        obj_mat = self.obj_info()

        obj_mat[0, :] = obj_mat[0, :] - ego_s  # 障碍车辆与自车的纵向相对距离
        obj_sorted_id = np.argsort(abs(obj_mat[0, :]))  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        obj_mat_surr = obj_mat[:, obj_sorted_id][:, 0:8]

        for i in range(np.shape(obj_mat_surr)[1]):
            state_vector[(i + 1) * 3] = obj_mat_surr[0][i] / self.d_max_s
            state_vector[(i + 1) * 3 + 1] = obj_mat_surr[2][i] / self.obj_max_vs
            df_obj = closest([self.LANE_WIDTH * lane_n for lane_n in range(-1, 3)], obj_mat_surr[1][i])
            state_vector[(i + 1) * 3 + 2] = (self.df_ego - df_obj) / 3.5 * (1 / 2)

        return state_vector

    def replay_ego_trajectory(self,data,t):
        # 回放自车轨迹
        num_other_vehicle = cfg.TRAFFIC_MANAGER.N_SPAWN_CARS
        x = data[t][0]
        y = data[t][num_other_vehicle + 1]
        z = data[t][2 * (num_other_vehicle + 1)]
        velocity_x = data[t][3 * (num_other_vehicle + 1)]
        velocity_y = data[t][4 * (num_other_vehicle + 1)]
        velocity_z = data[t][5 * (num_other_vehicle + 1)]
        pitch = data[t][6 * (num_other_vehicle + 1)] * 180 / 3.14159265358979323
        roll = data[t][7 * (num_other_vehicle + 1)] * 180 / 3.14159265358979323
        yaw = data[t][8 * (num_other_vehicle + 1)] * 180 / 3.14159265358979323

        transform_ego = carla.Transform(location=carla.Location(x=x, y=y, z=z),
                                        rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
        self.ego.set_transform(transform_ego)  # 重置主车位置

    def replay_other_trajectory(self,data,t):
        # 回放周车轨迹
        num_other_vehicle = cfg.TRAFFIC_MANAGER.N_SPAWN_CARS
        for i, actor in enumerate(self.traffic_module.actors_batch):  # 重置周车位置
            x = data[t][i + 1]
            y = data[t][(num_other_vehicle + 1) + i + 1]
            z = data[t][2 * (num_other_vehicle + 1) + i + 1]
            velocity_x = data[t][3 * (num_other_vehicle + 1) + i + 1]
            velocity_y = data[t][4 * (num_other_vehicle + 1) + i + 1]
            velocity_z = data[t][5 * (num_other_vehicle + 1) + i + 1]
            pitch = data[t][6 * (num_other_vehicle + 1) + i + 1] * 180 / 3.14159265358979323
            roll = data[t][7 * (num_other_vehicle + 1) + i + 1] * 180 / 3.14159265358979323
            yaw = data[t][8 * (num_other_vehicle + 1) + i + 1] * 180 / 3.141592653589793223


            transform_other = carla.Transform(location=carla.Location(x=x, y=y, z=z),
                                              rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
            actor['Actor'].set_transform(transform_other)

    def initialize_ego_state(self,data,t):
        # 回放自车初始状态
        num_other_vehicle = cfg.TRAFFIC_MANAGER.N_SPAWN_CARS
        x = data[t][0]
        y = data[t][num_other_vehicle + 1]
        z = data[t][2 * (num_other_vehicle + 1)]
        v_x = data[t][3 * (num_other_vehicle + 1)]
        v_y = data[t][4 * (num_other_vehicle + 1)]
        v_z = data[t][5 * (num_other_vehicle + 1)]
        pitch = data[t][6 * (num_other_vehicle + 1)] * 180 / 3.14159265358979323
        roll = data[t][7 * (num_other_vehicle + 1)] * 180 / 3.14159265358979323
        yaw = data[t][8 * (num_other_vehicle + 1)] * 180 / 3.14159265358979323

        transform_ego = carla.Transform(location=carla.Location(x=x, y=y, z=z),
                                        rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
        self.ego.set_transform(transform_ego)  # 重置主车位置
        self.ego.set_simulate_physics(True)
        self.ego.set_target_velocity(carla.Vector3D(x=v_x, y=v_y, z=v_z))

    def initialize_other_state(self,data,t):
        # 回放周车初始状态
        num_other_vehicle = cfg.TRAFFIC_MANAGER.N_SPAWN_CARS
        for i, actor in enumerate(self.traffic_module.actors_batch):  # 重置周车位置
            x = data[t][i + 1]
            y = data[t][(num_other_vehicle + 1) + i + 1]
            z = data[t][2 * (num_other_vehicle + 1) + i + 1]
            v_x = data[t][3 * (num_other_vehicle + 1) + i + 1]
            v_y = data[t][4 * (num_other_vehicle + 1) + i + 1]
            v_z = data[t][5 * (num_other_vehicle + 1) + i + 1]
            pitch = data[t][6 * (num_other_vehicle + 1) + i + 1] * 180 / 3.14159265358979323
            roll = data[t][7 * (num_other_vehicle + 1) + i + 1] * 180 / 3.14159265358979323
            yaw = data[t][8 * (num_other_vehicle + 1) + i + 1] * 180 / 3.141592653589793223

            transform_other = carla.Transform(location=carla.Location(x=x, y=y, z=z),
                                              rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
            actor['Actor'].set_transform(transform_other)
            actor['Actor'].set_simulate_physics(True)
            actor['Actor'].set_target_velocity(carla.Vector3D(x=v_x, y=v_y, z=v_z))

    def step(self, action):
        ''' Initialize Flags '''
        collision = False
        init_steps = 0
        replay_mode = cfg.OTHER_INFOMATION.REPLAY_MODE
        ''' Critical Scenario Replay '''
        t = self.n_step + self.init_t
        '''在仿真的第一步，按照记录数据初始化自车和周车的初始状态'''
        if self.n_step == 2:
            self.initialize_ego_state(self.data, t)
            self.initialize_other_state(self.data, t)

        '''回放场景有两种方式：回放轨迹，回放策略'''
        # 回放轨迹
        if replay_mode == 2:
            if t < np.shape(self.data)[0]-1:
                self.replay_ego_trajectory(self.data, t)
                self.replay_other_trajectory(self.data, t)
            if t >= np.shape(self.data)[0]-1:
                print("记录片段回放结束")
                self.done = True
        if replay_mode == 3:
            if t < np.shape(self.data)[0]-1:
                self.replay_other_trajectory(self.data, t)
            if t >= np.shape(self.data)[0]-1:
                print("记录片段回放结束")
                self.done = True

        ''' Spactator Setting '''
        spectator = self.world_module.world.get_spectator()
        transform = self.ego.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(x=-5.5, y=0, z=2.5), carla.Rotation(pitch=-15, yaw=0)))

        ''' Select Adversarial BV '''
        # idx_bv_old = self.idx_bv
        # self.idx_bv = update_adversarial_bv(self.traffic_module.actors_batch, self.ego.get_location().x,self.ego.get_location().y,self.idx_bv)
        self.idx_bv = -1

        ''' Select Nearest BV '''
        # 从自车前后五十米的范围内选择距离自车最近的四辆车并返回其状态，idx1-4代表最近车辆的序号，idxx= -1代表没有车辆
        # idx1, idx2, idx3, idx4 = select_nearest_bv(self.traffic_module.actors_batch,self.ego.get_location().x,self.ego.get_location().y,self.idx_bv)

        ''' Observation Design '''
        # others_s = np.zeros(self.N_SPAWN_CARS)
        # others_d = np.zeros(self.N_SPAWN_CARS)
        # others_v_S = np.zeros(self.N_SPAWN_CARS)
        # others_v_D = np.zeros(self.N_SPAWN_CARS)
        # others_phi_Frenet = np.zeros(self.N_SPAWN_CARS)
        # others_x = np.zeros(self.N_SPAWN_CARS)
        # others_y = np.zeros(self.N_SPAWN_CARS)
        # others_vx = np.zeros(self.N_SPAWN_CARS)
        # others_vy = np.zeros(self.N_SPAWN_CARS)
        # others_psi = np.zeros(self.N_SPAWN_CARS)
        #
        # for i, actor in enumerate(self.traffic_module.actors_batch):
        #     # 获取字典中Obj-frenet_state中的车辆状态信息
        #     act_s, act_d, act_v_S, act_v_D, act_psi_Frenet, other_info = actor['Obj_Frenet_state']
        #     x_bv, y_bv, _, vx_bv, vy_bv, _, _, _, psi_bv, _, _, _ = actor["Obj_Cartesian_state"]
        #     others_s[i] = act_s
        #     others_d[i] = act_d
        #     others_v_S[i] = act_v_S
        #     others_v_D[i] = act_v_D
        #     others_phi_Frenet[i] = act_psi_Frenet
        #     others_x[i] = x_bv
        #     others_y[i] = y_bv
        #     others_vx[i] = vx_bv
        #     others_vy[i] = vy_bv
        #     others_psi[i] = psi_bv

        ''' Observation for Adversarial Background Vehicle '''
        # 获取自车状态
        # av_info = [self.ego.get_location().x, self.ego.get_location().y, self.ego.get_velocity().x,
        #            self.ego.get_velocity().y, math.radians(self.ego.get_transform().rotation.yaw)]
        #
        # if self.idx_bv == -1:
        #     ad_bv_info =[0,0,0,0,0]
        # else:
        #     ad_bv_info = [others_x[self.idx_bv]-av_info[0], others_y[self.idx_bv]-av_info[1], others_vx[self.idx_bv]-av_info[2], others_vy[self.idx_bv]-av_info[3],others_psi[self.idx_bv]-av_info[4]]
        #
        # if idx1 == -1:
        #     bv1_info = [0,0,0,0,0]
        # else:
        #     bv1_info = [others_x[idx1]-av_info[0], others_y[idx1]-av_info[1], others_vx[idx1]-av_info[2], others_vy[idx1]-av_info[3],others_psi[idx1]--av_info[4]]
        #
        # if idx2 == -1:
        #     bv2_info = [0,0,0,0,0]
        # else:
        #     bv2_info = [others_x[idx2]-av_info[0], others_y[idx2]-av_info[1], others_vx[idx2]-av_info[2], others_vy[idx2]-av_info[3],others_psi[idx2]--av_info[4]]
        #
        # if idx3 == -1:
        #     bv3_info = [0,0,0,0,0]
        # else:
        #     bv3_info = [others_x[idx3]-av_info[0], others_y[idx3]-av_info[1], others_vx[idx3]-av_info[2], others_vy[idx3]-av_info[3],others_psi[idx3]--av_info[4]]
        #
        # if idx4 == -1:
        #     bv4_info = [0,0,0,0,0]
        # else:
        #     bv4_info = [others_x[idx4]-av_info[0], others_y[idx4]-av_info[1], others_vx[idx4]-av_info[2], others_vy[idx4]-av_info[3],others_psi[idx4]--av_info[4]]
        #
        # obs_bv = av_info + ad_bv_info + bv1_info + bv2_info + bv3_info + bv4_info

        '''Calculate Action of BV'''
        # obs_bv = np.array(obs_bv).reshape(1, 30)
        # model_bv = SAC.load("./Model/BV_Model.zip")
        # action_bv = model_bv.predict(obs_bv)
        action_bv = -1
        '''Autonomous Driving System'''
        control_av, df_n, acc_input, ego_d = self.AV_action_to_control(action)

        '''Infomation'''
        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, self.max_s]

        vx_ego = self.ego.get_velocity().x
        vy_ego = self.ego.get_velocity().y
        vz_ego = self.ego.get_velocity().z
        ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]  # estimated current ego_s
        v_S, v_D = velocity_inertial_to_frenet(ego_s, vx_ego, vy_ego, self.motionPlanner.csp)

        '''Update Carla World'''
        if self.n_step == 1:  # 第一步，更新30步，让所有车辆进入高速行驶状态
            for i in range(init_steps):
                self.ego.apply_control(control_av)  # apply control
                for i, actor in enumerate(self.traffic_module.actors_batch):
                    actor['Actor'].apply_control(control_av)
                self.module_manager.tick([-1, 0])
        else:
            self.ego.apply_control(control_av)
            new_action = [self.idx_bv, action_bv]

            self.module_manager.tick(new_action)  # Update carla world

        '''Collision Sensor'''
        if self.auto_render:
            self.render()
        self.world_module.collision_sensor.reset()
        collision_hist = self.world_module.get_collision_history()
        if any(collision_hist):
            collision = True

        '''******   Reward for Autonmous Driving Vehicle   ******'''
        # # 碰撞惩罚
        if collision:
            reward_cl = -200.0
        else:
            reward_cl = 0.0
        # if check_collision:
        #     reward_cl += -5.0

        # # 速度优先在某范围
        scaled_speed_l = lamp(v_S, [0, self.minSpeed], [0, 1])  # v_S/v_min
        scaled_speed_h = lamp(v_S, [self.minSpeed, self.maxSpeed], [0, 1])  # (v_S-v_min)/(v_max-v_min)
        reward_hs_l = 0.5
        reward_hs_h = 4.0
        reward_sp = reward_hs_l * np.clip(scaled_speed_l, 0, 1) + reward_hs_h * np.clip(scaled_speed_h, 0, 1)

        # # 在车道线范围内
        distance_lc = abs(ego_d - self.df_ego) / 1.75
        reward_lc = -1.0
        reward_distance = reward_lc * distance_lc

        # # 超车奖励
        obj_info = self.obj_info()
        overtake = ego_s - obj_info[0, :] > 0
        num_overtake = np.sum(overtake)
        dnum = num_overtake - self.last_num_overtake
        reward_ot = 0.5
        reward_overtake = reward_ot * dnum

        # # 舒适性奖励
        reward_cf = -0.5
        reward_comfort = reward_cf * (abs(df_n - self.last_df_n) / 7.0 + abs(acc_input - self.last_acc_input) / 10.0)
        # reward_comfort = 0.0

        # # 总奖励
        reward_av = float(reward_cl + reward_sp + reward_distance + reward_overtake + reward_comfort)
        reward_av = reward_av/100
        self.episode_total_reward += reward_av
        # nearest vehicle also to far
        # if collision or min(obj_info[0, :]) - ego_s > 100.0 or max(obj_info[0, :]) - ego_s < -5.0 or ego_s > 2850 or collision == True or y_av > 45 :
        if  ego_s > 2850 or collision or max(obj_info[0, :]) - ego_s < -5.0:
            self.done = True


        obj_info_mux = self.obj_info()
        obj_info_mux = list(np.array(obj_info_mux).flatten())
        # print("Object_information:",obj_info_mux)

        #info = obj_info_mux
        info = {'reserved': 0}
        self.n_step += 1

        if self.done == True:
            self.num_episode += 1
            # self.episode_buffer.append(self.num_episode)
            if collision:
                self.num_collision += 1
            collision_rate = self.num_collision / self.num_episode * 100
            self.total_reward += self.episode_total_reward
            average_reward = self.total_reward / self.num_episode
            print('===============关键场景任务================')
            print(f'num_collision:{self.num_collision}')
            print(f'num_episode:{self.num_episode}')
            print(f'碰撞率：{collision_rate}（%）')
            print(f'本回合奖励：{self.episode_total_reward}')
            print(f'截至目前平均奖励：{average_reward}')
            print('=============================================')

        return self.obs_nearest_five_bv(), reward_av, self.done, info

    def reset(self):
        self.vehicleController.reset()
        self.PIDLongitudinalController.reset()#vehicleController中已经对两个方向的控制器进行了重置，这里是不是重复了？？？
        self.PIDLateralController.reset()
        self.world_module.reset()

        self.init_s = self.world_module.init_s
        self.init_d = self.world_module.init_d
        self.traffic_module.reset(self.init_s, self.init_d)
        self.motionPlanner.reset(self.init_s, self.init_d, df_n=0, Tf=4, Vf_n=0, optimal_path=False)
        self.last_acc_input = 0.0
        self.f_idx = 0
        self.done  = False
        self.n_step = 0  # initialize episode steps count
        self.eps_rew = 0
        self.is_first_path = True
        self.episode_total_reward = 0
        self.num_using_rules = 0
        self.num_using_RL = 0

        # Ego starts to move slightly after being relocated when a new episode starts. Probably, ego keeps a fraction of previous acceleration after
        # being relocated. To solve this, the following procedure is needed.
        # self.ego.set_simulate_physics(False)

        self.module_manager.tick(self.INIT_ACTION)#初始状动作
        # self.ego.set_simulate_physics(False)
        # 环境已经reset了，为什么不直接返回actors_batch??


        return np.zeros_like(self.observation_space.sample()[0, :])
        #return [self.traffic_module.actors_batch,0.0,0.0,0.0,0.0,0.0,0.0] #拼凑一个环境的观察的格式返回去

    def begin_modules(self, args):
        # define and register module instances

        self.module_manager = ModuleManager()# 用于modules的统一管理，如register,start,tick,reset等
        width, height = [int(x) for x in args.carla_res.split('x')]# 渲染的视频尺寸

        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager,
                                        width=width, height=height)#生成世界，包含ADS的初始位置设置，（还没生成ADS）

        self.traffic_module = TrafficManager(MODULE_TRAFFIC, module_manager=self.module_manager)# 生成Traffic Manager

        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.traffic_module)

        if args.play_mode:
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)

        #生成global route，数据在ModuleWorld初始化时从CARLA中的地图数据获得
        if self.global_route is None:
            self.global_route = np.empty((0, 3))
            distance = 1
            for i in range(1520):
                wp = self.world_module.town_map.get_waypoint(carla.Location(x=406, y=-100, z=0.1),
                                                             project_to_road=True).next(distance=distance)[0]
                distance += 2
                self.global_route = np.append(self.global_route,
                                              [[wp.transform.location.x, wp.transform.location.y,
                                                wp.transform.location.z]], axis=0)
                # To visualize point clouds
                self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
            self.global_route = np.vstack([self.global_route, self.global_route[0, :]])
            np.save('global_route_town04', self.global_route)
            # plt.plot(self.global_route[:, 0], self.global_route[:, 1])
            # plt.show()

        self.motionPlanner = MotionPlanner() # MotionPlanner为ADS生成轨迹用于侧向控制，还要用于对NPC的侧向控制，要用两次

        # Start Modules
        self.motionPlanner.start(self.global_route) # 更新global_route
        # solve Spline
        self.world_module.update_global_route_csp(self.motionPlanner.csp) # 更新global_route
        self.traffic_module.update_global_route_csp(self.motionPlanner.csp) # 更新global_route
        self.module_manager.start_modules()
        # self.motionPlanner.reset(self.world_module.init_s, self.world_module.init_d)

        self.ego = self.world_module.hero_actor
        self.actors_batch = self.traffic_module.actors_batch
        self.ego_los_sensor = self.world_module.los_sensor
        self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})
        self.PIDLongitudinalController = PIDLongitudinalController(self.ego, K_P=40.0, K_D=0.1, K_I=4.0)
        self.PIDLateralController = PIDLateralController(self.ego, K_P=1.5, K_D=0.0, K_I=0.0)
        self.IDM = IntelligentDriverModel(self.ego)

        self.module_manager.tick(self.INIT_ACTION)  # Update carla world，将初始action设为-1
        self.init_transform = self.ego.get_transform()

    def enable_auto_render(self):
        self.auto_render = True

    def render(self, mode='human', close=False):
        self.module_manager.render(self.world_module.display)

    def destroy(self):
        print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
            self.traffic_module.destroy()



