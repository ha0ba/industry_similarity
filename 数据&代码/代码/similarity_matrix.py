# -*- coding: utf-8 -*-
"""
@Author : Zijia Du
@ID     : ha0ba
@Mail   : shduzijia@sjtu.edu.cn
@Time   : 2021/6/15 22:23
@Version: PyCharm
"""

import numpy as np
import time


def string_to_int(x):
    return int(x)


def turn_to_matrix(year):
    # 在对称位置依次填入相似度
    origin_file_path = r'E:\similarity_files\result_' + str(year) + '.csv'   # 相似度文件位置
    origin_list = [each.strip().split(',') for each in open(origin_file_path, 'r', encoding='utf-8-sig')]
    print(origin_list[:5])
    firm_list = list(set([each[0] for each in origin_list]) | set([each[1] for each in origin_list]))
    firm_list.sort(key=string_to_int)   # 对firm_list进行排序，建造有序列
    firm_number = len(firm_list)
    firm_id = range(0, firm_number)
    print(firm_number)
    firm_id_dic = dict(zip(firm_list, firm_id))

    sim_matrix = np.zeros([firm_number + 1, firm_number + 1])
    for each in origin_list:
        if float(each[2]) != 0:
            sim_matrix[firm_id_dic[each[0]] + 1][firm_id_dic[each[1]] + 1] = float(each[2])
            sim_matrix[firm_id_dic[each[1]] + 1][firm_id_dic[each[0]] + 1] = float(each[2])

    for i in range(0, firm_number):
        sim_matrix[0][i + 1] = firm_list[i]
        sim_matrix[i + 1][0] = firm_list[i]



    np.savetxt(r'E:\similarity_matrix\matrix_' + str(year) + '.csv'
               , sim_matrix
               , encoding='utf-8', delimiter=',', newline='\n', fmt='%0.8g')


start = time.time()
for year in range(2010, 2018):
    turn_to_matrix(year)
print(time.time() - start)