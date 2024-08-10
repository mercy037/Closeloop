import os
import pandas as pd
import csv
import numpy as np
import re


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


class data_collection:

    def __init__(self):

        self.mpc_record = pd.DataFrame(columns=('u', 'du', 'Cur_acc', 'q', 'ru', 'rdu'))
        self.vehicle_info_record = pd.DataFrame(columns=('vs', 'vs_cmd', 'throttle'))
        self.vehicle_info_2_v_record = pd.DataFrame(columns=('ego_s', 's_ahead， vs, v_ahead'))

        self.directory_path = self.mkdir(os.path.abspath('.') + '/datas/plot_data_record/data_record_no_1')
        self.file_name_1 = 'mpc_data_record.csv'
        self.file_name_2 = 'vehicle_info_record.csv'
        self.file_name_3 = 'vehicle_info_2_v_record.csv'

    def data_record(self, info, name):
        if name == "mpc":
            self.mpc_record = self.mpc_record.append(
                pd.DataFrame({info[0]: [info[1]], info[2]: [info[3]], info[4]: [info[5]], info[6]: [info[7]], info[8]: [info[9]], info[10]: [info[11]]}),
                ignore_index=True)
            self.mpc_record.to_csv(self.directory_path + '/' + self.file_name_1,
                                   index=False)
        elif name == "vehicle_info":
            self.vehicle_info_record = self.vehicle_info_record.append(
                pd.DataFrame({info[0]: [info[1]], info[2]: [info[3]], info[4]: [info[5]]}), ignore_index=True)
            self.vehicle_info_record.to_csv(self.directory_path + '/' + self.file_name_2,
                                            index=False)
        elif name == "vehicle_info_2_v":
            self.vehicle_info_2_v_record = self.vehicle_info_2_v_record.append(
                pd.DataFrame({info[0]: [info[1]], info[2]: [info[3]], info[4]: [info[5]], info[6]: [info[7]]}), ignore_index=True)
            self.vehicle_info_2_v_record.to_csv(self.directory_path + '/' + self.file_name_3,
                                            index=False)

    def mkdir(self, path):
        """
        创建指定的文件夹
        :param path: 文件夹路径，字符串格式
        :return: True(新建成功) or False(文件夹已存在，新建失败)
        """

        path = path.strip()
        path = path.rstrip("\\")
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            return path
        else:
            path = self.directory_check(path)
            os.makedirs(path)
            return path

    @staticmethod
    def directory_check(directory_check):
        temp_directory_check = directory_check
        i = 1
        while i:

            if os.path.exists(temp_directory_check):
                search = '_'
                numList = [m.start() for m in re.finditer(search, temp_directory_check)]
                numList[-1]
                temp_directory_check = temp_directory_check[0:numList[-1] + 1] + str(i)
                i = i + 1
            else:
                return temp_directory_check
