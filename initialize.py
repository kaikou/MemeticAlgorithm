#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)

import GeneticAlgorithm as ga
import random
from decimal import Decimal
import numpy as np
import pandas as pd

# # 遺伝子情報の長さ
# GENOM_LENGTH = 50
# 遺伝子集団の長さ
MAX_GENOM_LIST = 30
# 各両親から生成される子個体の数
MAX_CHILDREN = 10
# # 遺伝子選択数
# SELECT_GENOM = 20
# # 個体突然変異確率
# INDIVIDUAL_MUTATION = 0.1
# # 遺伝子突然変異確率
# GENOM_MUTATION = 0.1
# 繰り返す世代数
MAX_GENERATION = 30
# 使用できる車両数
VEHICLE = 3
# 車両の最大積載量
CAPACITY = 40


"""
避難所情報のデータフレームを生成する
@INPUT:
    filepath: 読み出すファイルパス
    data_name: 読み出すファイル名
@OUTPUT:
    読み出したデータのデータフレーム
"""
def createDataFrame(filepath, data_name):
    input_path = filepath + data_name + ".csv"
    return pd.read_csv(input_path)


"""
避難所の距離に基づいたコスト行列を返す
@INPUT:
    num_shelter : 避難所数
@OUTPUT:
    arr : コスト行列
"""
def createCostMatrix(num_shelter):
    dis = []
    arr = np.empty((0, num_shelter), int) #小数点以下を加える→float型

    for i in range(num_shelter):
        for j in range(num_shelter):
            x_crd = df.ix[j].x - df.ix[i].x
            y_crd = df.ix[j].y - df.ix[i].y

            dis.append(int(np.sqrt(np.power(x_crd, 2) + np.power(y_crd, 2))))
            if j == num_shelter - 1:
                arr = np.append(arr, np.array([dis]), axis=0)
                dis = []

    print(df[0:11])
    print("コスト行列-----------------------------------")
    print(arr)
    print("---------------------------------------------")
    np.savetxt("./output/cost.csv", arr, delimiter=',', fmt='%.2f')
    return arr


def isNearDepot(edge, x):
    if x[edge[0]][0] or x[0][edge[0]]:
        if x[edge[1]][0] or x[0][edge[1]]:
            return 1
    return 0



def savingMethod(num_shelter, cost):
    print(cost)
    saving = np.zeros((num_shelter, num_shelter), int) #小数点以下を加える→float型
    # s_x = np.zeros((num_shelter, num_shelter), int) #小数点以下を加える→float型
    # s_arr = np.empty((0, 3), int)
    s = np.zeros((num_shelter, num_shelter), int)
    # route_flag = False
    # selected = []

    q = np.zeros((num_shelter), int)
    nex = np.zeros((num_shelter), int)
    tail = np.zeros((num_shelter), int)
    dr = np.zeros((num_shelter), int)

    for i in range(1, num_shelter):
        # s_x[0][i] = 1
        # s_x[i][0] = 1

        q[i] = df.ix[i].d
        nex[i] = 0
        tail[i] = i
        dr[i] = cost[0][i] + cost[i][0]


    """
    セービングファイルを作成する
    s_arrは，

    [i  j  s]
    [i  j  s]
    ......
    という構造になっており，
    i-j間のセービング値がsであり，sについて降順にソートしている
    """
    # for i in range(1, num_shelter):
    #     for j in range(i+1, num_shelter):
    #         saving[i][j] = cost[i][0] + cost[0][j] - cost[i][j]
    #         if saving[i][j] > 0:
    #             s_arr = np.append(s_arr, np.array([[i, j, saving[i][j]]]), axis=0)
    # s_arr = s_arr[s_arr[:, 2].argsort()][::-1]
    # # print(saving)
    # print(s_arr)

    for i in range(1, num_shelter):
        for j in range(i, num_shelter):
            saving[i][j] = cost[i][0] + cost[0][j] - cost[i][j]
            s[i][j] = saving[i][j]
    # print(saving)
    print(s)


    smax = 1
    while(smax != 0):
        smax = 0
        for i in range(1, num_shelter):
            if q[i] > 0:
                for j in range(1, num_shelter):
                    if i != j and q[j] > 0:
                        ti = tail[i]
                        if q[i] + q[j] <= CAPACITY and s[ti][j] > smax:
                            smax = s[ti][j]
                            g = i
                            h = j
        if smax > 0:
            nex[tail[g]] = h
            tail[g] = tail[h]
            dr[g] = dr[g] + dr[h] - smax
            q[g] = q[g] + q[h]
            q[h] = 0

    print(nex)


    route = []
    heiro = []

    drt = 0
    for i in range(1, num_shelter):
        if q[i] > 0:
            ii = i
            while(True):
                heiro.append(ii)
                ii = nex[ii]
                if ii == 0:
                    heiro.append(dr[i]) # その経路の移動コスト
                    heiro.append(q[i]) # その経路の総需要
                    drt += dr[i]
                    route.append(heiro)
                    heiro = []
                    break

    print(route)




if __name__ == "__main__":
    filename = "data_r101"

    df = createDataFrame("./data/", filename)
    # num_shelter = len(df.index)
    num_shelter = 11

    # 各避難所間の移動コスト行列を生成する
    # 2次元配列costで保持
    cost = createCostMatrix(num_shelter)

    print(df[:11])
    # print(cost)

    savingMethod(num_shelter, cost)
