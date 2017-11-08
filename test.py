#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)

import GeneticAlgorithm as ga
import random
from decimal import Decimal
import numpy as np
import pandas as pd
import random


# 遺伝子情報の長さ
GENOM_LENGTH = 50
# 遺伝子集団の長さ
MAX_GENOM_LIST = 10
# 遺伝子選択数
SELECT_GENOM = 20
# 個体突然変異確率
INDIVIDUAL_MUTATION = 0.1
# 遺伝子突然変異確率
GENOM_MUTATION = 0.1
# 繰り返す世代数
MAX_GENERATION = 2
# 使用できる車両数
VEHICLE = 4


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



def createGenom(num_shelter, m):
    genom_list = []
    # ルート区切りナンバーの数は車両数-1
    genom_list = [i for i in range(1, num_shelter + m - 1)]
    random.shuffle(genom_list)
    #genom_list = genom_list[::-1] #逆順
    #return genom_list
    return ga.genom(genom_list, 0)

def setRondomOrder(genom_list):
    random_order_list = []
    order = [i for i in range(MAX_GENOM_LIST)]
    random.shuffle(order)

    for j in order:
        random_order_list.append(genom_list[j])
    print("ランダムな並び")
    print(np.array(random_order_list))



if __name__ == '__main__':



    filepath = "./data/" #ファイルパス
    df = pd.read_csv(filepath + "data_r101.csv") #読み出す避難所の位置情報ファイル
    #num_shelter = len(df.index)
    num_shelter = 11

    # 各避難所間の移動コスト行列を生成する
    # 2次元配列costで保持
    cost = createCostMatrix(num_shelter)

    # 第一世代の個体集団を生成
    current_generation_individual_group = []
    print("現行の個体集団")
    for i in range(MAX_GENOM_LIST):
        current_generation_individual_group.append(createGenom(num_shelter, VEHICLE))
    genom_list = np.array(current_generation_individual_group)
    print(genom_list)



    setRondomOrder(genom_list)
