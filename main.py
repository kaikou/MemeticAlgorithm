#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)
#
# 配送経路問題をMemeticアルゴリズムを用いて解くプログラム
# Memeticプログラムは遺伝的アルゴリズムと局所探索を組合せた手法である
#
# GeneticAlgorithm.pyは遺伝子情報とその遺伝子の評価値を格納するclass
# 値を取得する場合は.getGenom()

import GeneticAlgorithm as ga
import random
from decimal import Decimal
import numpy as np
import pandas as pd
import random

# 遺伝子情報の長さ
GENOM_LENGTH = 50
# 遺伝子集団の長さ
MAX_GENOM_LIST = 5
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

"""
評価関数．
巡回路の移動コストが低い方が良い解となる．
@INPUT：
    ga : 評価を行うgenomClass
@OUTPUT:
    total_cost : 評価処理をしたgenomClass
"""
def evaluation(ga):
    # 配送順序の配列を変数genomにコピー
    genom = ga.getGenom()[:]
    route_cost = 0
    total_cost = 0
    route_flag = False
    for i in range(len(genom)):
        # ルート区切り番号だった場合
        if genom[i] > num_shelter - 1: # >10
            if route_flag == True:
                route_cost += cost[genom[i-1]][0]
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
                route_flag = True
                print("{}→{}".format(0, genom[i]))
            else : # フラグがTrue，つまり経路続行
                route_cost += cost[genom[i-1]][genom[i]]
                print("{}→{}".format(genom[i-1], genom[i]))
    # 遺伝子の最後の番号が区切り番号でない場合，
    # 避難所から配送拠点までの移動コストを加算する
    if route_flag == True:
        total_cost += route_cost + cost[genom[i]][0]
        print("{}→{}".format(genom[i], 0))
    print("総移動コスト:{}".format(total_cost))
    print("-------------経路------------")
    return total_cost

# """
# 【エリート選択】
# 評価値が優れている順にソートを行った後，
# 一定以上の遺伝子を抽出する
# @INPUT:
#     ga : 選択を行うgenomClassの配列
#     elite : 遺伝子選択数
# @OUTPUT:
#     選択処理をした一定のエリートgenomClass
# """
# def select(ga, elite):
#     # 現行世代個体集団の評価値を高い順にソートする
#     sort_result = sorted(ga, reverse=True, key=lambda u: u.evaluation)
#     # 一定の上位を抽出する
#     result = []


"""
集団中の個体をランダムな順序に並べるための個体番号順列を生成する
@INPUT:
    None
@OUTPUT:
    ランダムに並べた遺伝子集団サイズ分の順列
"""
def setRondomOrder():
    #random_order_list = []
    # 集団サイズ分の順列リストを作り，シャッフルする
    random_order = [i for i in range(MAX_GENOM_LIST)]
    random.shuffle(random_order)
    return(random_order)





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
        print(current_generation_individual_group[i].getGenom())
    """"
    ここまで第一世代
    この先繰り返し
    """
    for count_ in range(1, MAX_GENERATION + 1):
        #集団中の個体をランダムな順列に並べる
        order = setRondomOrder()
        print("ランダムな順列" + str(order))

        #現行世代個体集団の遺伝子を評価し，genomClassに代入
        for i in range(MAX_GENOM_LIST):
            evaluation_result = evaluation(current_generation_individual_group[i])
            current_generation_individual_group[i].setEvaluation(evaluation_result)

        #遺伝子集団それぞれの評価値確認
        print("====第{}世代====".format(count_))
        for i in range(MAX_GENOM_LIST):
            print("遺伝子<{}>:{}".format(i + 1, current_generation_individual_group[i].getEvaluation()))

        #エリート個体を選択する
        #elite_genes = select(current_generation_individual_group, SELECT_GENOM)
