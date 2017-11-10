#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)

import GeneticAlgorithm as ga
import numpy as np
import pandas as pd
import random

VEHICLE = 3


"""
避難所の距離に基づいたコスト行列を返す
@INPUT:
    num_shelter : 避難所数
@OUTPUT:
    arr : コスト行列
"""
def createCostMatrix(num_shelter, df):
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

def createDataFrame(data_name):
    filepath = "./data/" #ファイルパス
    df = pd.read_csv(filepath + data_name + ".csv") #読み出す避難所の位置情報ファイル
    return df

"""
引数で指定された避難所数と車両数に基づき，
避難所番号の順列及びルート区切りによる
ランダムな個体を生成，格納したgenomClassで返す．
@INPUT:
    num_shelter : 避難所数
    m : 車両数
@OUTPUT:
    生成した個体集団genomClass
"""
def createGenom(num_shelter, m):
    genom_list = []
    # ルート区切りナンバーの数は車両数-1
    genom_list = [i for i in range(1, num_shelter + m - 1)]
    random.shuffle(genom_list)
    #genom_list = genom_list[::-1] #逆順
    return ga.genom(genom_list, 0)



def evaluation(ga, num_shelter, cost):
    # 配送順序の配列を変数genomにコピー
    genom = ga.getGenom()[:]
    route_cost = 0
    total_cost = 0
    route_flag = False
    aiueo = 0
    x = np.zeros((num_shelter, num_shelter), int) #小数点以下を加える→float型
    for i in range(len(genom)):
        # ルート区切り番号だった場合
        if genom[i] > num_shelter - 1: # >10
            if route_flag == True:
                route_cost += cost[genom[i-1]][0]
                x[genom[i-1]][0] = 1
                print("{}→{}".format(genom[i-1], 0))
            total_cost += route_cost
            route_cost = 0
            route_flag = False
            print("_{}_区切り".format(genom[i]))
        else : # ルート区切り番号ではない場合(避難所番号)

            # 現在参照している避難所番号の前が区切り番号だった，
            # もしくは遺伝子の最初を参照している場合
            if route_flag == False:
                # 配送拠点から避難所までの移動コストを加算
                route_cost += cost[0][genom[i]]
                x[0][genom[i]] = 1
                route_flag = True
                print("{}→{}".format(0, genom[i]))
            else : # フラグがTrue，つまり経路続行
                route_cost += cost[genom[i-1]][genom[i]]
                x[genom[i-1]][genom[i]] = 1
                print("{}→{}".format(genom[i-1], genom[i]))
    # 遺伝子の最後の番号が区切り番号でない場合，
    # 避難所から配送拠点までの移動コストを加算する
    if route_flag == True:
        total_cost += route_cost + cost[genom[i]][0]
        x[genom[i]][0] = 1
        print("{}→{}".format(genom[i], 0))
    print("総移動コスト:{}".format(total_cost))
    for i in range(num_shelter):
        for j in range(num_shelter):
            aiueo += cost[i][j] * x[i][j]
    print("総移動コスト2:{}".format(aiueo))
    print("-------------経路------------")
    print(x)
    return total_cost
