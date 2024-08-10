import math
import numpy as np

'''
完成修改
'''


class Config:

    def __init__(self):
        """
        Simulation env Config
        """

        self.N = 5000  # 共走多少步， 注意此时仿真步长为0.1s，假设X超过20m时开始变道，D_ref变为lane_width
        self.T = 0.1
        self.eps = 1e-10

        '''
        Vehicle Config
        '''

        self.v_x_0 = 0 / 3.6
        self.v_y_0 = 0.0 / 3.6
        self.Pos_x_0 = 1010.0

        self.a = 1.200
        self.b = 1.40
        self.L = self.a + self.b
        self.M = 1650
        self.Iz = 3234
        self.Caf = -50531 * 2
        self.Car = -43121 * 2
        self.R = 0.353  # 轮胎半径
        self.steeringratio = 16  # 方向盘转16度，前轮转1度

        '''
        Display Config
        '''
        self.sim_result_record_enable = True


class MPC_Config_0(Config):

    def __init__(self):
        super().__init__()

        self.dstop = 10
        self.K_ref = 0
        self.omega_ref = 0
        self.mpc_Nx = 3  # State Size
        self.mpc_Nu = 2  # Input Size
        self.mpc_Ny = 3
        self.mpc_Nw = 1  # 噪声数目：1
        self.mpc_Np = 60
        self.mpc_Nc = 30
        self.mpc_Row = 10
        self.mpc_Cy = [[1, 0], [0, 1]]
