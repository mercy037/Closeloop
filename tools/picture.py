#!/bin/bash python

import csv
import numpy as np
import xlrd
from matplotlib import pyplot as plt
import os
import pandas as pd


def exact_data(f):
    col_types = [float]
    f_csv = csv.reader(f)
    data = []
    index = 1
    for col in f_csv:
        col = tuple(convert(value) for convert, value in zip(col_types, col))
        data.append(col[0])
        index += 1

    return np.array(data)


def plot_data(rec_num, name_1="mpc", name_2="u", name_3="s_ahead"):
    if name_1 == "mpc":
        if isinstance(rec_num, int):

            record_num = 'data_record_no_' + str(rec_num)
            d1 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num + '/mpc_data_record.csv')

            u = d1.u.values
            cur_acc = d1.Cur_acc.values
            q = d1.q.values
            ru = d1.ru.values
            rdu = d1.rdu.values
            du = d1.du.values

            t = np.arange(0, u.shape[0])
            plt.figure()
            plt.plot(t[0:-1], u[0:-1], color="blue", label='u')
            plt.plot(t[0:-1], cur_acc[0:-1], color="red", label="cur_acc")
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], q[10:-1], color="green", label="q")
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], ru[10:-1], color="blue", label="ru")
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], rdu[10:-1], color="black", label="rdu")
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], du[10:-1], color="orange", label="du")
            plt.legend()

            plt.show()

        else:

            record_num_1 = 'data_record_no_' + str(rec_num[0])
            record_num_2 = 'data_record_no_' + str(rec_num[1])
            d1 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num_1 + '/mpc_data_record.csv')
            d2 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num_2 + '/mpc_data_record.csv')

            u1 = eval("d1." + name_2 + ".values")
            u2 = eval("d2." + name_2 + ".values")

            t1 = np.arange(0, u1.shape[0])
            t2 = np.arange(0, u2.shape[0])
            plt.figure()
            plt.plot(t1[10:-1], u1[10:-1], color="blue", label=str(record_num_1) + "_" + name_2)
            plt.plot(t2[10:-1], u2[10:-1], color="red", label=str(record_num_2) + "_" + name_2)
            plt.legend()
            plt.show()
    elif name_1 == "vehicle_info":

        if isinstance(rec_num, int):

            record_num = 'data_record_no_' + str(rec_num)
            d1 = pd.read_csv(
                os.path.abspath('..') + "/datas/plot_data_record/" + record_num + '/vehicle_info_record.csv')

            vs = d1.vs.values
            vs_cmd = d1.vs_cmd
            throttle = d1.throttle

            t = np.arange(0, vs.shape[0])
            plt.figure()
            plt.plot(t[10:-1], vs[10:-1], color="blue", label='vs')
            plt.legend()

            plt.plot(t[10:-1], vs_cmd[10:-1], color="red", label="vs_cmd")
            plt.legend()

            plt.figure()
            plt.plot(t[10:-1], throttle[10:-1], color="red", label="throttle")
            plt.show()

        else:

            record_num_1 = 'data_record_no_' + str(rec_num[0])
            record_num_2 = 'data_record_no_' + str(rec_num[1])
            d1 = pd.read_csv(
                os.path.abspath('..') + "/datas/plot_data_record/" + record_num_1 + '/vehicle_info_record.csv')
            d2 = pd.read_csv(
                os.path.abspath('..') + "/datas/plot_data_record/" + record_num_2 + '/vehicle_info_record.csv')

            u1 = eval("d1." + name_2 + ".values")
            u2 = eval("d2." + name_2 + ".values")

            t1 = np.arange(0, u1.shape[0])
            t2 = np.arange(0, u2.shape[0])
            plt.figure()
            plt.plot(t1[0:-1], u1[0:-1], color="blue", label=str(record_num_1))
            plt.plot(t2[0:-1], u2[0:-1], color="red", label=str(record_num_2))
            plt.legend()
            plt.show()
    else:

        if isinstance(rec_num, int):

            record_num = 'data_record_no_' + str(rec_num)
            d1 = pd.read_csv(
                os.path.abspath('..') + "/datas/plot_data_record/" + record_num + '/vehicle_info_2_v_record.csv')

            ego_s = d1.ego_s.values
            s_ahead = d1.s_ahead
            vs = d1.vs.values
            v_ahead = d1.v_ahead
            R_s = s_ahead - ego_s
            R_v = v_ahead - vs
            t = np.arange(0, ego_s.shape[0])
            plt.figure()
            plt.plot(t[10:-1], R_s[10:-1], color="blue", label='R_s')
            plt.legend()
            plt.figure()
            plt.plot(t[10:-1], R_v[10:-1], color="red", label='R_v')
            plt.legend()

            plt.show()

        else:

            record_num_1 = 'data_record_no_' + str(rec_num[0])
            record_num_2 = 'data_record_no_' + str(rec_num[1])
            d1 = pd.read_csv(
                os.path.abspath('..') + "/datas/plot_data_record/" + record_num_1 + '/vehicle_info_2_v_record.csv')
            d2 = pd.read_csv(
                os.path.abspath('..') + "/datas/plot_data_record/" + record_num_2 + '/vehicle_info_2_v_record.csv')

            u11 = eval("d1." + name_2 + ".values")
            u12 = eval("d1." + name_3 + ".values")
            u1 = u12 - u11
            u21 = eval("d2." + name_2 + ".values")
            u22 = eval("d2." + name_3 + ".values")
            u2 = u22 - u21
            t1 = np.arange(0, u1.shape[0])
            t2 = np.arange(0, u2.shape[0])
            plt.figure()
            plt.plot(t1[0:-1], u1[0:-1], color="blue", label=str(record_num_1))
            plt.plot(t2[0:-1], u2[0:-1], color="red", label=str(record_num_2))
            plt.legend()
            plt.show()


if __name__ == '__main__':
    '''
    rec_num = [1, 2]

    '''
    rec_num = [189,188]
    # rec_num = 169
    '''
    ego_s,  s_ahead， vs, v_ahead
    '''
    plot_data(rec_num, 'vehicle_info_2_v', 'ego_s','s_ahead')
    plot_data(rec_num, 'vehicle_info_2_v', 'vs','v_ahead')
    '''
    vs, vs_cmd, throttle
    '''
    plot_data(rec_num, 'vehicle_info', 'vs_cmd')
    plot_data(rec_num, 'vehicle_info', 'throttle')
    '''
    u, du, Cur_acc, q, ru, rdu
    '''
    plot_data(rec_num, 'mpc', 'du')
    plot_data(rec_num, 'mpc', 'u')
    plot_data(rec_num, 'mpc', 'q')

# t = np.arange(0, 1000)
# ob = xlrd.open_workbook('/home/wangliwen/Carla_RL_MPC/tools/ob_v_data.xlsx')
# ob = ob.sheets()[0]#第几个sheet
# plt.plot(t,np.array(ob.row_values(0)),'g',label='ob_v_data')
# plt.show()
