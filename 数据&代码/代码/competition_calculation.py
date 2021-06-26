# -*- coding: utf-8 -*-
"""
@Author : Zijia Du
@ID     : ha0ba
@Mail   : shduzijia@sjtu.edu.cn
@Time   : 2021/6/19 21:09
@Version: PyCharm
"""

import numpy as np
import time


start = time.time()


def write(year):
    # 构造对应年份的相似度阈值，高于阈值即为行业对
    year_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    threshold_list = [0.570393756, 0.583129422, 0.591050415, 0.602143734, 0.607857082, 0.581847576, 0.567988652, 0.535743053]
    # origin threshold_list
    # threshold_list = [0.5938081250165151, 0.6076460957587665, 0.6188393183048388, 0.6215006416904897, 0.631950518549322, 0.6038428997838268, 0.5964012236245932, 0.5680695106338924]
    # zx threshold_list
    # threshold_list = [0.4483374249255523, 0.456586492718257, 0.474663310657965, 0.48760972131528524, 0.4945457412346193, 0.46144669304893526, 0.44585178846659324, 0.433931502586702]
    # 0.2_0.5_coefficient
    threshold_dic = dict(zip(year_list, threshold_list))
    threshold = threshold_dic.get(year)
    # financial_dic 为公司的比较信息字典，键为stkcd，值为对应公司的五个比较数值

    similarity_path = r'E:\similarity_matrix\matrix_' + str(year) + '.csv'   # 相似度矩阵文件位置
    similarity_list = [each.strip().split(',') for each in open(similarity_path, 'r', encoding='utf-8-sig')]
    similarity_firm_list = similarity_list[0][1:]
    similarity_index_list = range(len(similarity_firm_list))
    similarity_index_dic = dict(zip(similarity_index_list, similarity_firm_list))
    # similarity_index_dic 为公司的位置字典，键为对应相似度的位置，值为对应公司的stkcd
    sim_firm_list_2 = [each[0] for each in similarity_list[1:]]
    sim_sim_list = [each[1:] for each in similarity_list[1:]]
    sim_dic = dict(zip(sim_firm_list_2, sim_sim_list))
    # sim_dic 为相似度字典，键为公司stkcd，值为其与其他公司的相似度
    firm_num = len(similarity_firm_list)
    # 对目标公司做循环
    result_path = r'E:\computation\competition.csv'
    f = open(result_path, 'a+', encoding='utf-8')

    for each in similarity_firm_list:
        if sim_dic.get(each) is None:
            print(each)

    for each in similarity_firm_list:
        similarity_extent = 0
        tmp_sim = sim_dic.get(each)
        for i in range(0, firm_num):
            if float(tmp_sim[i]) > threshold:
                similarity_extent += float(tmp_sim[i])
        f.write(','.join([each, str(year)]) + ',' + str(similarity_extent) + '\n')
# 2544
# 300240


for z in range(2010, 2018):
    write(z)
