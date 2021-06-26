# -*- coding: utf-8 -*-
"""
@Author : Zijia Du
@ID     : ha0ba
@Mail   : shduzijia@sjtu.edu.cn
@Time   : 2021/6/13 03:17
@Version: PyCharm
"""

import numpy as np
import time
from scipy.optimize import linear_sum_assignment


# 延续原来想法，总的相似度 = 产品相似度 * 结构相似度
# 先算产品相似度，需要进行匹配。对m*n矩阵先选 m个不同行不同列产品, 剩下n-m列选最大的匹配就行。 (m<=n)
# m个不同行不同列直接用 linear_sum_assignment库完成。 最后生成n对产品序号及标识号，计算产品相似度，同时返回产品序号用于计算结构相似度
# 利用产品序号及标识号，计算结构相似度。


def compute_product_similarity(firm_list_1, firm_list_2):
    """
    firm_list: [code_list, ratio_list]
    匹配最相近的产品
    返回：最相近的产品对序号及标识号（用于找到最后 n - m个产品), 以及产品相似度
    """
    # 先调用make_mat 计算每个产品之间的相似度
    initial_product_similarity = make_mat(firm_list_1[0], firm_list_2[0])
    # 再调用linea_sum_assignment 找到m个最相近的产品
    # symbol_list 表示这一对是原来m个还是剩下的n-m个
    pair_num = min(len(firm_list_1[0]), len(firm_list_2[0]))
    row_index, col_index = linear_sum_assignment(initial_product_similarity, maximize=True)
    symbol_list = np.ones([pair_num], dtype=int)
    product_pair = np.array(list(zip(row_index, col_index, symbol_list)))
    # 再找剩下n-m个产品中相似度最大的值
    # 分类讨论
    if len(firm_list_1[0]) > len(firm_list_2[0]):
        for each in range(0, len(firm_list_1[0])):
            if each not in product_pair[:, 0]:
                tmp_list = initial_product_similarity[each, :].tolist()
                tmp_position = tmp_list.index(max(tmp_list))
                product_pair = np.append(product_pair, [[each, tmp_position, 0]], axis=0)
    elif len(firm_list_1[0]) < len(firm_list_2[0]):
        for each in range(0, len(firm_list_2[0])):
            if each not in product_pair[:, 1]:
                tmp_list = initial_product_similarity[:, each].tolist()
                tmp_position = tmp_list.index(max(tmp_list))
                product_pair = np.append(product_pair, [[tmp_position, each, 0]], axis=0)
    sum_numerator = 0
    sum_denominator = 0
    for each in product_pair:
        sum_numerator += float(firm_list_1[1][each[0]]) * float(firm_list_2[1][each[1]])\
                         * initial_product_similarity[each[0]][each[1]]
        sum_denominator += float(firm_list_1[1][each[0]]) * float(firm_list_2[1][each[1]])
    product_similarity = sum_numerator / sum_denominator

    return product_similarity, product_pair


def compute_structure_similarity(firm_list_1, firm_list_2, product_pair):
    """
    利用产品对序号及标识号，计算结构相似度
    返回：结构相似度
    """
    tmp_firm_1_struct = []
    tmp_firm_2_struct = []
    if len(firm_list_1[0]) >= len(firm_list_2[0]):
        for each in product_pair:
            tmp_firm_1_struct.append(float(firm_list_1[1][each[0]]))
            tmp_firm_2_struct.append(float(firm_list_2[1][each[1]]) * each[2])
    else:
        for each in product_pair:
            tmp_firm_1_struct.append(float(firm_list_1[1][each[0]]) * each[2])
            tmp_firm_2_struct.append(float(firm_list_2[1][each[1]]))
    firm_1_struct = np.array(tmp_firm_1_struct)
    firm_2_struct = np.array(tmp_firm_2_struct)
    structure_similarity = np.dot(firm_1_struct, firm_2_struct)\
                           / (np.linalg.norm(firm_1_struct) * np.linalg.norm(firm_2_struct))

    return structure_similarity


def make_mat(code1, code2):
    """
    :param code1: 产品列表1, list
    :param code2: 产品列表2, list
    :return: 不同产品之间的相似度矩阵
    0.7表示2级相似， 0.3表示1级相似
    """
    num1 = len(code1)
    num2 = len(code2)
    mat = np.zeros((num1, num2), dtype=float)

    for i in range(num1):
        for j in range(num2):
            if code1[i] == code2[j]:
                mat[i][j] = 1
            elif code1[i][:3] == code2[j][:3]:
                mat[i][j] = 0.7
            elif code1[i][:2] == code2[j][:2]:
                mat[i][j] = 0.3
    return mat


def compute_firm_similarity(firm_list_1, firm_list_2, firm_list):
    """
    传入两个公司的产品及比例 list   [code_list, ratio_list]
    :param firm_list_1: firm 1
    :param firm_list_2: firm 2
    :return: 最终总的公司产品相似度
    """
    product_similarity, product_pair = compute_product_similarity(firm_list_1, firm_list_2)
    structure_similarity = compute_structure_similarity(firm_list_1, firm_list_2, product_pair)
    final_similarity = product_similarity * structure_similarity
    final_result = firm_list[:]
    final_result.append(str(final_similarity))

    return final_result


def write_result(final_result):
    if final_result is not None:
        with open(result_path, 'a+', encoding='utf-8') as f1:
            f1.write(','.join(final_result) + '\n')


if __name__ == '__main__':
    start = time.time()
    file_path = r'E:\origin_files\2010.csv'   # 产品编码及收入占比文件
    result_path = r'E:\similarity_files\result_2010.csv'
    file_list = [each.strip().split(',') for each in open(file_path, 'r', encoding='utf-8-sig')][1:]
    product_number = int(len(file_list[0]) / 2)
    firm_dic = {}
    firm_list = [each[0] for each in file_list]
    for each in file_list:
        tmp_ratio = list(filter(None, each[1:product_number + 1]))
        tmp_code = list(filter(None, each[-product_number:]))
        firm_dic[each[0]] = np.array([tmp_code, tmp_ratio])
    del file_list

    with open(result_path, 'a+', encoding='utf-8') as f:
        for p in range(0, 50):   # len(firm_list)
            for q in range(p + 1, 50):
                tmp_result = compute_firm_similarity(firm_dic[firm_list[p]], firm_dic[firm_list[q]], [firm_list[p], firm_list[q]])
                f.write(','.join(tmp_result) + '\n')

    print('running time is ', time.time() - start)
    #   without pool running time is  101.55691933631897

'''x1 = [['29101', '19201'], ['30', '70']]
x2 = [['29101', '19201'], ['70', '30']]
print(compute_product_similarity(x1, x2))
print(compute_firm_similarity(x1, x2, ['x1', 'x2']))'''