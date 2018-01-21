#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)

import GeneticAlgorithm as ga
import random
from decimal import Decimal
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys, time
import copy
import csv
import math

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
m = 3
# 車両の最大積載量
CAPACITY = 160
# セービング値の効果をコントロールする係数
LAMBDA = 1
# N_near()関数で，どこまで近くのノードに局所探索するか
NEAR = 10
# penaltyFunction()で，容量制約違反に課すペナルティの係数
ALPHA = 1
# penaltyFunction()で，経路長違反に課すペナルティの係数
BETA = 1.0
# penaltyFunction()で，経路長違反とする距離
D = float("inf")
SERVICE = 0

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
    arr = np.empty((0, num_shelter), float) #小数点以下を加える→float型

    for i in range(num_shelter):
        for j in range(num_shelter):
            x_crd = df.ix[j].x - df.ix[i].x
            y_crd = df.ix[j].y - df.ix[i].y

            dis.append(round(np.sqrt(np.power(x_crd, 2) + np.power(y_crd, 2)), 2))
            if j == num_shelter - 1:
                arr = np.append(arr, np.array([dis]), axis=0)
                dis = []
    # print(df[0:11])
    # print("コスト行列-----------------------------------")
    # print(arr)
    # print("---------------------------------------------")
    np.savetxt("./output/cost.csv", arr, delimiter=',', fmt='%.2f')
    return arr


"""
セービング法によりヒューリスティックな解を生成する
@INPUT:
    num_shelter : 避難所数
    cost : 避難所間のコスト行列
@OUTPUT:

"""
def savingMethod(num_shelter, cost):
    """
    各避難所間のコストからセービング値を計算
    経路結合のための初期化をする
    """
    s = np.zeros((num_shelter, num_shelter), int)
    q = np.zeros((num_shelter), int)
    nex = np.zeros((num_shelter), int)
    tail = np.zeros((num_shelter), int)
    dr = np.zeros((num_shelter), int)

    for i in range(1, num_shelter):
        q[i] = df.ix[i].d
        nex[i] = 0
        tail[i] = i
        dr[i] = cost[0][i] + cost[i][0]

    for i in range(1, num_shelter):
        # for j in range(i+1, num_shelter):
        for j in range(1, num_shelter):
            s[i][j] = cost[i][0] + cost[0][j] - (LAMBDA * cost[i][j])
    # print(s) #セービング値

    """
    経路の結合処理
    """
    random_order1 = [i for i in range(1, num_shelter)]
    random.shuffle(random_order1)
    random_order2 = [i for i in range(1, num_shelter)]
    random.shuffle(random_order2)

    # smax = 1
    while(True):
        smax = 0
        # for i in range(1, num_shelter): #1から順に
        for i in random_order1: # ランダムにノードを選ぶ
            if q[i] > 0:
                # for j in range(1, num_shelter):
                for j in random_order2:
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

        if smax == 0:
            break

    """
    経路の出力
    """
    route = []
    heiro = []
    demand = []
    distance = []
    # plot = []
    drt = 0
    for i in random_order1:
        if q[i] > 0:
            ii = i
            while(True):
                heiro.append(ii)
                # plot.append(heiro)
                # graphPlot(pathToRoute(createEdgeSet(plot)), isFirst=0, isLast=0)

                ii = nex[ii]
                if ii == 0:
                    distance.append(dr[i]) # その経路の移動コスト
                    demand.append(q[i]) # その経路の総需要
                    drt += dr[i]
                    route.append(heiro)
                    heiro = []
                    # graphPlot(pathToRoute(createEdgeSet(route)), isFirst=0, isLast=0)
                    break

    # print("route:{}".format(route))
    print("demand:{}".format(demand))
    print("distance:{}".format(distance))
    print(drt)
    # graphPlot(pathToRoute(createEdgeSet(route)), isFirst=0, isLast=1)
    return route


    """
    セービングファイルを作成する
    s_arrは，

    [i  j  s]
    [i  j  s]
    ......
    という構造になっており，
    i-j間のセービング値がsであり，sについて降順にソートしている
    """
    # s_x = np.zeros((num_shelter, num_shelter), int) #小数点以下を加える→float型
    # s_arr = np.empty((0, 3), int)
    #
    # for i in range(1, num_shelter):
    #     for j in range(i+1, num_shelter):
    #         saving[i][j] = cost[i][0] + cost[0][j] - cost[i][j]
    #         if saving[i][j] > 0:
    #             s_arr = np.append(s_arr, np.array([[i, j, saving[i][j]]]), axis=0)
    # s_arr = s_arr[s_arr[:, 2].argsort()][::-1]
    # # print(saving)
    # print(s_arr)


"""
セービング法で構築されたルートに，
デポを出発してデポに帰るようにエッジを辿る
3次元配列を生成する．
@INPUT:
    route:セービング法で構築されたルート(0は表示されていない)
@OUTPUT:
    path:ルートを辿るエッジの3次元配列(0も表示)
"""
def createEdgeSet(route):
    E = []
    Path = []
    for edge in route:
        for i, node in enumerate(edge):
            if i == 0:
                E.append([0, node])
                pre_node = node
                if len(edge) == 1:
                    E.append([node, 0])
            else:
                E.append([pre_node, node])
                pre_node = node
                if i == len(edge)-1:
                    E.append([node, 0])
        Path.append(E)
        E = []
    # print(Path)
    return Path


"""
ペナルティ関数による評価を行う
@INPUT:
    route: 解の２次元リスト
    option: どのように評価するか
        2: 容量超過量F_cのみ
        1: ペナルティ関数による評価
        0: 総移動コストのみの評価
@OUTPUT:
    F_p: 関数による評価値
"""
def penaltyFunction(route, option):
    F_p = 0
    F = 0
    F_c = 0
    F_d = 0
    R_demands = 0
    R_cost = 0
    service = -1*SERVICE
    path = routeToPath(route)

    # ルート総距離
    for e in route:
        F += cost[int(e[0])][int(e[1])]

    if option == 0:
        return round(F, 2)

    for edges in path:

        """
        容量制約違反の計算
        """
        nodes = np.unique(edges)
        for n in nodes: # 各ルートの合計需要
            R_demands += df.ix[n].d
            service += SERVICE

        if R_demands > CAPACITY:
            F_c += R_demands - CAPACITY # ルート内の需要超過
        else:
            F_c += 0
            # F_c += abs(R_demands - CAPACITY) # ルート内の需要超過
        R_demands = 0

        """
        route duration制約違反の計算
        """
        for e in edges:
            R_cost += cost[int(e[0])][int(e[1])]
        R_cost += service
        if R_cost > D:
            F_d += R_cost - D
        else:
            F_d += 0
            # F_d += abs(R_cost - D)
        R_cost = 0
        service = -1*SERVICE

    # ペナルティ関数

    F_p = F + (ALPHA * F_c) + (BETA * F_d)
    # F_p = F + (ALPHA * F_c)
    # print("F:{}, F_c:{}, F_d:{}".format(F, F_c, F_d))
    if option == 1:
        return round(F_p, 2) # ペナルティ間数値
    else:
        return round(F_c + F_d, 2) # ペナルティ項のみ



"""
経路を分割することでルート数を増やす関数
セービング法によって生成された初期解のうち，
もっとも顧客数の多いルートを選び真ん中で分割する
@INPUT:
    path
    (m): 固定したいルート数
@OUTPUT:
    path
"""
def routeSplit(path):
    routeLength = []
    routeCost = []
    routeCapa = []
    s_option = 1
    # 0: ランダム
    # 1: エッジ数
    # 2: 距離
    # 3: 需要量
    # print("パス:{}".format(path))
    while(True):
        for route in path:
            r_cost = 0
            r_demand = 0
            routeLength.append(len(route)) # ルート毎の顧客数リスト
            for n in np.unique(route):
                r_demand += df.ix[n].d
            for e in route:
                r_cost += cost[int(e[0])][int(e[1])]
            routeCapa.append(r_demand) # ルート毎の需要量
            routeCost.append(round(r_cost, 2)) # ルート毎の距離リスト

        # print("routeLength:{}".format(routeLength))
        # print("routeCapa:{}".format(routeCapa))
        # print("routeCost:{}".format(routeCost))

        # エッジ数4以上のところからランダムに選ぶ
        if s_option == 0:
            IdxList = [i for i, x in enumerate(routeLength) if x >= 4]
            print(IdxList)
        # エッジ数の多いルートを選ぶ
        elif s_option == 1:
            IdxList = [i for i, x in enumerate(routeLength) if x == max(routeLength)]
        # ルート内の距離が一番長いルートを選ぶ
        elif s_option == 2:
            IdxList = [i for i, x in enumerate(routeCost) if x == max(routeCost)]
        # ルート内の需要が一番多いルートを選ぶ
        elif s_option == 3:
            IdxList = [i for i, x in enumerate(routeCapa) if x == max(routeCapa)]

        maxIdx = random.choice(IdxList)
        # print("maxIdx:{}".format(maxIdx))
        # print("それぞれの顧客数:{}".format(routeLength))

        # 分割するルート
        split = path[maxIdx]
        # 分割するルートを除いたパス
        path.pop(maxIdx)
        # print("split:{}".format(split))
        # print("残ったパス:{}".format(path))
        # 分割するインデックス
        Idx = math.floor(len(split)/2 - 1)
        # print("Idx:{}".format(Idx))

        #分岐点となるノードを決定する
        pointA = split[Idx][1] if split[Idx][1] in split[Idx] and split[Idx][1] in split[Idx+1] \
        else split[Idx][0]
        pointB = split[Idx+1][1] if split[Idx+1][0] == pointA else split[Idx+1][0]

        routeA = split[:Idx+1] + [[pointA, 0]]
        routeB = [[0, pointB]] + split[Idx+2:]

        # print("routeA:{}".format(routeA))
        # print("routeB:{}".format(routeB))

        path.append(routeA)
        path.append(routeB)

        # print("分割後パス:{}".format(path))

        routeLength = []

        if len(path) >= m:
            break
    return path

"""
あるノードから近いノード集合を返す
引数nodeで与えたノードから，近くにあるノードをnear番目まで選んだ集合
@INPUT:
    node:ノード
    near:近くのノードをいくつ探すか
@OUTPUT:
    near_cost:近くのノード集合
"""
def N_near(node, near):
    near_cost = np.empty((0, 2), int)
    for i, c in enumerate(cost[int(node)][:]):
        if i == 0:
            continue
        near_cost = np.append(near_cost, np.array([[i, c]]), axis=0)
    near_cost = near_cost[near_cost[:, 1].argsort()] # nodeに近い順にソート
    # print(near_cost)

    # print(near_cost[1:near+1, 0])
    return near_cost[1:near+1, 0]


def localSearch(path):
    # EdgeSet = []
    # for edge in path:
    #     for j in edge:
    #         EdgeSet.append(j)

    # 顧客ノードの部分集合
    List = [i for i in range(1, num_shelter)]
    random.shuffle(List) # ランダムな順に並べる
    print(List)

    for i in range(1, len(List)):
        v = List[i]


"""
近傍操作関数
@INPUT:
    v: 近傍操作対象ノード
    path: 解の3次元リスト
    f_option: ペナルティ関数をどのように評価するか
        2: 容量超過
        1: ペナルティ関数による評価
        0: 総移動コストのみの評価
    reduce_route: 経路数が減る遷移を許容するか
        1: 許容する
        0: 許容しない
"""
def Neighborhoods(v, path, f_option, reduce_route):
    EdgeSet = []
    # DefaultEdgeSet = []
    path_cost = 0
    bestSet = []
    flag = False

    # 解を構成する全てのエッジ集合を生成
    for edge in path:
        for j in edge:
            EdgeSet.append(j)
    # DefaultEdgeSet = copy.deepcopy(EdgeSet)
    bestSet = copy.deepcopy(EdgeSet)
    # print("デフォ:{}".format(DefaultEdgeSet))

    # 元の解のペナルティ関数評価値を保持
    P_eval = penaltyFunction(bestSet, f_option)
    # print("デフォルト:{}".format(P_eval))

    # 渡されたノードvに繋がるエッジ2つ
    link_v = [i for i in EdgeSet if (v in i)]
    v_minus = link_v[0][0] if v == link_v[0][1] else link_v[0][1]
    v_plus = link_v[1][1] if v == link_v[1][0] else link_v[1][0]
    # print("link_v:{}".format(link_v))
    # print("link_v[0]:{}".format(link_v[0])) # ノードvに向かうエッジ
    # print("link_v[1]:{}".format(link_v[1])) # ノードvを出るエッジ

    # ノードvからnearだけ近いノードをそれぞれwとして選ぶ
    for w in N_near(v, NEAR):
        w = int(w)
        link_w = [i for i in EdgeSet if (w in i)]

        if(len(link_v) < 2 or len(link_w) < 2):
            # print("ノードに対するエッジが2つない")
            EdgeSet = copy.deepcopy(bestSet)
            continue

        # 同じエッジを選ぶ可能性があるものは排除
        # if link_v[0] == link_w[0] or link_v[1] == link_w[0] or \
        # link_v[0] == link_w[1] or link_v[1] == link_w[1]:
        #     continue

        # ノードwに向かうエッジ
        w_minus = link_w[0][0] if w == link_w[0][1] else link_w[0][1]
        # ノードwを出るエッジ
        w_plus = link_w[1][1] if w == link_w[1][0] else link_w[1][0]
        # print("w:" + str(w))
        # print("w-:{}".format(link_w[0]))
        # print("w+:{}".format(link_w[1]))
        # print("link_w[0]:{}".format(link_w[0]))
        # print("link_w[1]:{}".format(link_w[1]))

        """
        (1,0)Interchange
        """
        # if neighbor == "10inter":
        # print("(1,0)Interchange①やるよー")
        """
        (1,0)Interchange①
        """
        # 選んだwに対して
        # 元々繋がっているエッジ
        l1 = cost[w_minus][w] # w-→w
        l2 = cost[v_minus][v] # v-→v
        l3 = cost[v][v_plus] # v→v+
        # # つなぎ直すエッジ
        l4 = cost[w_minus][v]
        l5 = cost[v][w]
        l6 = cost[v_minus][v_plus]

        # if l1 + l2 + l3 > l4 + l5 + l6:
        # print("10inter① " + str(v) + ":" + str(w) + "適用")
        try:
            EdgeSet.remove(link_w[0]) #-を含む方
            EdgeSet.remove(link_v[0])
            EdgeSet.remove(link_v[1]) #+を含む方

            EdgeSet.append([w_minus, v])
            EdgeSet.append([v, w])
            EdgeSet.append([v_minus, v_plus])
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    # ルート数が減る遷移をどうするか
                    if es[0] == 0 and reduce_route == 1:
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False
        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 + l3 > l4 + l5 + l6):
                        print("修正")
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)

            # ペナルティ関数値が元より大きい
            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet

        flag = False
        """
        (1,0)Interchange②
        """
        # 元々繋がっているエッジ
        l1 = cost[w][w_plus] # w→w+
        l2 = cost[v_minus][v] # v-→v
        l3 = cost[v][v_plus] # v→v+
        # つなぎ直すエッジ
        l4 = cost[w_plus][v]
        l5 = cost[v][w]
        l6 = cost[v_minus][v_plus]

        # if l1 + l2 + l3 > l4 + l5 + l6:
        # print("10inter② " + str(v) + ":" + str(w) + "適用")
        try:
            EdgeSet.remove(link_w[1]) #+を含む方
            EdgeSet.remove(link_v[0])
            EdgeSet.remove(link_v[1]) #+を含む方

            EdgeSet.append([w_plus, v])
            EdgeSet.append([v, w])
            EdgeSet.append([v_minus, v_plus])
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    if es[0] == 0 and reduce_route == 1:
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False
        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 + l3 > l4 + l5 + l6):
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)

            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet
        flag = False

        """
        (0,1)Interchange
        """
        # if neighbor == "01inter":
        # print("(0,1)Interchange①やるよー")
        """
        (0,1)Interchange①
        """
        # 選んだwに対して
        # 元々繋がっているエッジ
        l1 = cost[w_minus][w]
        l2 = cost[w][w_plus]
        l3 = cost[v_minus][v]
        # つなぎ直すエッジ
        l4 = cost[w_minus][w_plus]
        l5 = cost[v_minus][w]
        l6 = cost[v][w]

        # if l1 + l2 + l3 > l4 + l5 + l6:
        # print("01inter① " + str(v) + ":" + str(w) + "適用")
        try:
            EdgeSet.remove(link_w[0]) #-を含む方
            EdgeSet.remove(link_w[1])
            EdgeSet.remove(link_v[0]) #-を含む方

            EdgeSet.append([w_minus, w_plus])
            EdgeSet.append([v_minus, w])
            EdgeSet.append([v, w])
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    # ルート数が減る遷移をどうするか
                    if es[0] == 0 and reduce_route == 1:
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False

        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 + l3 > l4 + l4 + l5):
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)
            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet
        flag = False

        """
        (0,1)Interchange②
        """
        # 元々繋がっているエッジ
        l1 = cost[w_minus][w]
        l2 = cost[w][w_plus]
        l3 = cost[v][v_plus]
        # つなぎ直すエッジ
        l4 = cost[w_minus][w_plus]
        l5 = cost[v_plus][w]
        l6 = cost[v][w]

        # if l1 + l2 + l3 > l4 + l5 + l6:
        # print("01inter② " + str(v) + ":" + str(w) + "適用")
        try:
            EdgeSet.remove(link_w[0]) #-を含む方
            EdgeSet.remove(link_w[1])
            EdgeSet.remove(link_v[1]) #-を含む方

            EdgeSet.append([w_minus, w_plus])
            EdgeSet.append([v_plus, w])
            EdgeSet.append([v, w])
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1] and reduce_route == 1:
                    if es[0] == 0:
                        # ルート数が減る遷移をどうするか
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False

        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 + l3 > l4 + l5 + l6):
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)
            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet
        flag = False

        """
        (1,1)Interchange
        """
        # if neighbor == "11inter":
            # print("(1,1)Interchangeやるよー")
        """
        (1,1)Interchange①
        """

        """
        w--を求めるステップ
        """
        path = routeToPath(EdgeSet)
        # ノードwがどのルートに含まれるか求める
        for i, route in enumerate(path):
            for n in route:
                if w in n:
                    route_num = i
                    break
        # print("{}を含むルートは:{}".format(w, path[route_num]))
        # if(len(path[route_num]) < 3):
        #     continue

        # ノードw-を持つリストを作る
        w_minus2list = [x for x in path[route_num] \
                        if(w_minus in x and x != link_w[0])]

        node_flag = False
        try:
            # リストからノードw-でない方をノードw--として選択
            w_minus_minus = w_minus2list[0][0] if w_minus2list[0][1] == w_minus \
            else w_minus2list[0][1]
            node_flag = True
        except IndexError:
            print("w--作成で失敗")
            node_flag = False

        # 同じエッジを選ぶ可能性があるものは排除
        # if link_v[0] == w_minus2list[0] or link_v[1] == w_minus2list[0] \
        # or link_w[0] == w_minus2list[0] or link_w[1] == w_minus2list[0]:
        #     continue
        if node_flag == True:
            # 選んだwに対して
            # 元々繋がっているエッジ
            l1 = cost[w_minus_minus][w_minus]
            l2 = cost[w_minus][w]
            l3 = cost[v_minus][v]
            l4 = cost[v][v_plus]
            # つなぎ直すエッジ
            l5 = cost[w_minus_minus][v]
            l6 = cost[w_minus][v_plus]
            l7 = cost[v_minus][w_minus]
            l8 = cost[v][w]

            # if l1 + l2 + l3 + l4 > l5 + l6 + l7 + l8:
                # print("11inter① " + str(v) + ":" + str(w) + "適用")
            try:
                # print("w-のリスト:{}".format(w_minus2list[0]))
                # print("link_w[0]:{}".format(link_w[0]))
                # print("link_v[0]:{}".format(link_v[0]))
                # print("link_v[1]:{}".format(link_v[1]))
                EdgeSet.remove(w_minus2list[0]) #--を含む方
                EdgeSet.remove(link_w[0])
                EdgeSet.remove(link_v[0])
                EdgeSet.remove(link_v[1])

                EdgeSet.append([w_minus_minus, v])
                EdgeSet.append([w_minus, v_plus])
                EdgeSet.append([v_minus, w_minus])
                EdgeSet.append([v, w])
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1] and reduce_route == 1:
                        if es[0] == 0:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False

            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    # print("リセット1")
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 + l3 + l4 > l5 + l6 + l7 + l8):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)
                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    return bestSet
            flag = False

        """
        (1,1)Interchange②
        """

        """
        w++を求めるステップ
        """
        # ノードw+を持つリストを作る
        w_plus2list = [x for x in path[route_num] \
                        if(w_plus in x and x != link_w[1])]

        node_flag = False
        try:
            # リストからノードw+でない方をノードw++として選択
            w_plus_plus = w_plus2list[0][1] if w_plus2list[0][0] == w_plus \
            else w_plus2list[0][0]
            node_flag = True
        except IndexError:
            print("w++の作成に失敗")
            node_flag = False

        # print("{}を含むルートは:{}".format(w, path[route_num]))
        # print("リスト:{}".format(w_plus2list[0]))
        # print("w+:{}".format(w_plus))
        # print("w++:{}".format(w_plus_plus))
        if node_flag == True:
            # 元々繋がっているエッジ
            l1 = cost[w_plus][w_plus_plus]
            l2 = cost[w][w_plus]
            l3 = cost[v_minus][v]
            l4 = cost[v][v_plus]
            # つなぎ直すエッジ
            l5 = cost[v][w_plus_plus]
            l6 = cost[v_minus][w_plus]
            l7 = cost[v_plus][w_plus]
            l8 = cost[v][w]

            # 同じエッジを選ぶ可能性があるものは排除
            # if link_v[0] == w_plus2list[0] or link_v[1] == w_plus2list[0] \
            # or link_w[0] == w_plus2list[0] or link_w[1] == w_plus2list[0]:
            #     continue

            # if l1 + l2 + l3 + l4 > l5 + l6 + l7 + l8:
            # print("11inter② " + str(v) + ":" + str(w) + "適用")
            try:
                EdgeSet.remove(w_plus2list[0]) #++を含む方
                EdgeSet.remove(link_w[1])
                EdgeSet.remove(link_v[0])
                EdgeSet.remove(link_v[1])

                EdgeSet.append([v, w_plus_plus])
                EdgeSet.append([v_minus, w_plus])
                EdgeSet.append([v_plus, w_plus])
                EdgeSet.append([v, w])
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1] and reduce_route == 1:
                        if es[0] == 0:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False

            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    # print("リセット２")
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 + l3 + l4 > l5 + l6 + l7 + l8):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)
                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    return bestSet
            flag = False

        """
        2-opt近傍
        """
        # if neighbor == "2opt":
        """
        2-opt①
        """
        # print("2opt①やるよー")
        l1 = cost[v_minus][v] # 元のエッジ
        l2 = cost[w_minus][w] # 元のエッジ
        l3 = cost[v][w]
        l4 = cost[v_minus][w_minus]

        # if l1 + l2 > l3 + l4:
        # print("2-opt① " + str(v) + ":" + str(w) + "適用")
        try:
            EdgeSet.remove(link_w[0])
            EdgeSet.remove(link_v[0])
            EdgeSet.append([v, w])
            EdgeSet.append([v_minus, w_minus])
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            for es in EdgeSet:
                if es[0] == es[1]:
                    if es[0] == 0 and reduce_route == 1:
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False

        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 > l3 + l4):
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)
            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet
        flag = False
        """
        2-opt④
        """
        # print("2opt④やるよー")
        l1 = cost[v][v_plus] # 元のエッジ
        l2 = cost[w][w_plus] # 元のエッジ
        l3 = cost[v][w]
        l4 = cost[v_plus][w_plus]

        # if l1 + l2 > l3 + l4:
        # print("2-opt④ " + str(v) + ":" + str(w) + "適用")
        try:
            EdgeSet.remove(link_v[1])
            EdgeSet.remove(link_w[1])
            EdgeSet.append([v, w])
            EdgeSet.append([v_plus, w_plus])
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            for es in EdgeSet:
                if es[0] == es[1]:
                    if es[0] == 0 and reduce_route == 1:
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False

        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 > l3 + l4):
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)

            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet
        flag = False

        # vとwが同じルートか判定
        sameRoute = False
        path = routeToPath(EdgeSet)
        for r in path:
            node = np.unique(r)
            if v in node and w in node:
                sameRoute = True

        #ノードvとwが異なるルートに存在する時
        if not(sameRoute):
            """
            2-opt②
            """
            # print("2opt②やるよー")
            l1 = cost[v_minus][v] # 元のエッジ
            l2 = cost[w][w_plus] # 元のエッジ
            l3 = cost[v][w]
            l4 = cost[v_minus][w_plus]

            # if l1 + l2 > l3 + l4:
            # print("2-opt② " + str(v) + ":" + str(w) + "適用")
            try:
                EdgeSet.remove(link_v[0])
                EdgeSet.remove(link_w[1])
                EdgeSet.append([v, w])
                EdgeSet.append([v_minus, w_plus])
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                for es in EdgeSet:
                    if es[0] == es[1]:
                        if es[0] == 0 and reduce_route == 1:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False

            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 > l3 + l4):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)
                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    return bestSet
            flag = False
            """
            2-opt③
            """
            # print("2opt③やるよー")
            l1 = cost[v][v_plus] # 元のエッジ
            l2 = cost[w_minus][w] # 元のエッジ
            l3 = cost[v][w]
            l4 = cost[v_plus][w_minus]

            # if l1 + l2 > l3 + l4:
            # print("2-opt③ " + str(v) + ":" + str(w) + "適用")
            try:
                EdgeSet.remove(link_v[1])
                EdgeSet.remove(link_w[0])
                EdgeSet.append([v, w])
                EdgeSet.append([v_plus, w_minus])
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1]:
                        if es[0] == 0 and reduce_route == 1:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False
            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 > l3 + l4):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)

                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    return bestSet
    return bestSet


"""
2次元のエッジリストから，各閉路毎にエッジを持つ3次元リストに変換する
@INPUT:
    route: エッジの２次元リスト
@OUTPUT:
    Path: ルート情報を含むエッジの3次元リスト
    False: 巡回路を構築できない
"""
def routeToPath(route):
    route = sorted(route)
    Path = []
    R = []
    v_e_1 = 0

    while(len(route)):
        e = route[0]
        find_flag = False
        heiro = False
        if int(e[0]) == 0 and int(e[1]) == 0:
            route.remove(e)
            continue
        # エッジ端のどちらかに0を含むか
        elif int(e[0]) == 0 or int(e[1]) == 0:
            R.append(e)
            route.remove(e)
            # 0じゃない方をv_eにセット
            v_e = int(e[1]) if int(e[0]) == 0 else int(e[0])
        else: # 0を含まない閉路を発見
            R.append(e)
            route.remove(e)
            v_e = int(e[1])
            v_e_1 = int(e[0])
            heiro = True

        while(not(find_flag)):
            # v_eを含むエッジをroute内から探しeにセット
            if (v_e not in np.unique(route)):
                print("見つからない")
                return False
            e = random.choice(list(filter(lambda x: int(v_e) in x, route)))
            R.append(e)
            route.remove(e)
            # print("-------------------")
            # print("v_e:{}".format(v_e))
            # print("v_eを含むエッジ:{}".format(e))
            # print("route:{}".format(route))
            # print("Path:{}".format(Path))

            # eの端点のv_eでない方を新たにv_eとする
            v_e = int(e[0]) if v_e == int(e[1]) else int(e[1])
            # print("次の端点:{}".format(v_e))

            if(heiro == True):
                if(v_e == v_e_1):
                    Path.append(R)
                    v_e_1 = 0
                    R=[]
                    find_flag = True
            elif(v_e == 0):
                Path.append(R)
                R = []
                find_flag = True
    return(Path)



"""
3次元のエッジリストから，2次元リストに変換する
@INPUT:
    path: ルート情報を含むエッジの3次元リスト
@OUTPUT:
    route: エッジの2次元リスト
"""
def pathToRoute(path):
    EdgeList = []
    for edge in path:
        for j in edge:
            EdgeList.append(j)
    return EdgeList




"""
デポを含まない閉路が存在するかどうか判定し，
そのインデックスを返す
@INPUT:
    path: ルート情報を持つ3次元リスト
@OUTPUT:
    I: 部分巡回路のインデックスリスト
"""
def isHeiro(path):
    I = []
    for i, r in enumerate(path):
        # print(r)
        if 0 in np.unique(r):
            pass
        else:
            I.append(i)
    # print("部分巡回路のインデックス:{}".format(I))
    if I == []:
        return 0
    else:
        return I

"""
実行不可能解の修正操作
@INPUT:
    route: 解の2次元配列
@OUTPUT:
    modi_route: 修正後の解の2次元配列
"""
def modification(route):
    path = routeToPath(route)
    modi_route = []
    pena = False
    modicount = 0

    r, excess = checkCapacity(path)
    D_r, D_excess = checkDuration(path)

    if r == [] and D_r == []:
        return route, "success"
    elif excess == False:
        print("修正不可")
        return route,"error"


    while(True):
        while(r):
            ExIdx = random.choice(r)
            # print("超過ルートのインデックス:{}".format(ExIdx))
            print("実行不可能解あり")
            prePath = copy.deepcopy(path)

            for i in np.unique(path[ExIdx]):
                # print("i:{}".format(i))
                # print("np.unique(path[ExIdx]):{}".format(np.unique(path[ExIdx])))
                # sys.stdout.write("\r%d番目ルートの修正操作" % ExIdx)
                sys.stdout.flush()
                time.sleep(0.01)
                modi_route = Neighborhoods(i, path, f_option=2, reduce_route=0)

                path = routeToPath(modi_route)
                r, excess = checkCapacity(path)
                if(r == []):
                    pena = True
                    print("容量違反を抜けた")
                    break
            # print("ルートに対する修正操作後の評価関数値:{}".format(penaltyFunction(modi_route, option=2)))
            if pena == True:
                break
            elif prePath == path:
                print("【modification】修正失敗")
                # print(route)
                return route,"error"
            # r = []
            # r, excess = checkCapacity(path)

        """
        期間制約違反の評価
        """
        D_r, D_excess = checkDuration(path)
        while(D_r != []):
            pena = False

            ExIdx = random.choice(D_r)
            # print("超過ルートのインデックス:{}".format(ExIdx))
            print("実行不可能解あり")
            prePath = copy.deepcopy(path)

            for i in np.unique(path[ExIdx]):
                # print("i:{}".format(i))
                # print("np.unique(path[ExIdx]):{}".format(np.unique(path[ExIdx])))
                # sys.stdout.write("\r%d番目ルートの修正操作" % ExIdx)
                sys.stdout.flush()
                time.sleep(0.01)
                modi_route = Neighborhoods(i, path, f_option=2, reduce_route=0)

                path = routeToPath(modi_route)
                D_r, D_excess = checkDuration(path)
                if(D_r == []):
                    pena = True
                    break

            # print("ルートに対する修正操作後の評価関数値:{}".format(penaltyFunction(modi_route, option=2)))
            if pena == True:
                print("期間違反を抜けた")
                break
            elif prePath == path:
                print("【modification】修正失敗")
                # print(route)
                return route,"error"

        r, excess = checkCapacity(path)
        D_r, D_excess = checkDuration(path)
        print(r)
        print(D_r)

        if(r == [] and D_r == []):
            break
        modicount += 1
        if modicount == 10:
            return route, "error"
    # print(modi_route)
    return modi_route,"success"

"""
modification()中に用いる
入力された3次元リストを元に，トラックの容量超過をしているルートの
インデックスとルート毎の総需要量のリストを返す
"""
def checkCapacity(path):
    excess = []
    r = []
    r_demands = 0

    for i, heiro in enumerate(path):
        for n in np.unique(heiro):
            r_demands += df.ix[n].d
        demand = r_demands - CAPACITY\
         # if r_demands > CAPACITY else 0
        excess.append(demand)
        if(demand > 0):
            r.append(i)
        r_demands = 0
        demand = 0
    # print("容量違反インデックス:{}".format(r))
    # print("各容量違反:{}".format(excess))
    # ルート内の需要オーバーの合計が0を超えていたら
    if(sum(excess) > 0):
        return r, False
    else:
        return r, excess


def checkDuration(path):
    D_excess = []
    D_r = []
    r_duration = 0

    for i, heiro in enumerate(path):
        """
        route duration制約違反の計算
        """
        for e in heiro:
            r_duration += cost[int(e[0])][int(e[1])]

        duration = r_duration - D
        D_excess.append(int(duration))
        if duration > 0:
            D_r.append(i)
        r_duration = 0
        duration = 0
    print("期間違反インデックス:{}".format(D_r))
    # print("各期間違反:{}".format(D_excess))
    # ルート内の需要オーバーの合計が0を超えていたら

    return D_r, D_excess
"""
グラフをプロットする
"""
def graphPlot(edgeList, isFirst, isLast, title):
    # X = []
    # Y = []
    N = []
    G = nx.Graph()
    pos = {}  #ノードの位置情報格納

    # # デポ以外の座標を代入
    # for i in range(num_shelter):
    #     X.append(df.ix[i].x)
    #     Y.append(df.ix[i].y)

    # ノード番号とノードの座標を格納
    for i in range(num_shelter):
        N.append(i)
        pos[i] = (df.ix[i].x, df.ix[i].y)

    E = []
    edge_labels = {}
    sum_cost = 0
    labels = {}

    for e in edgeList:
        E.append(e)
        edge_labels[(int(e[0]), int(e[1]))] = int(cost[int(e[0])][int(e[1])])

    for i in range(num_shelter):
        # labels[i] = df.ix[i].d
        labels[i] = i

    G.add_nodes_from(N)
    G.add_edges_from(E)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="r")
    nx.draw_networkx_edges(G, pos, width=1)
    # nx.draw_networkx(G, pos, with_labels=False, node_color='r', node_size=80) # デフォルト200
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6) # デフォルト12
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6) # デフォルト8

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.xlim(0, 70)
    # plt.ylim(0, 70)
    # plt.axis('off')
    # plt.grid()

    # 元の経路
    if isFirst == 1:
        print("最初の経路:{}".format(penaltyFunction(edgeList, 0)))
        print("ペナルティ関数値:{}".format(penaltyFunction(edgeList, 1)))
        print("ルート数:{}".format(len(routeToPath(edgeList))))
        plt.title(title)
        plt.savefig("./output/param/ini_" + filename +".png")  # save as png
        plt.pause(0.01)
        plt.figure()

    # 連続プロット中
    if isLast == 0:
        plt.pause(0.01)
        plt.clf()
    else:
        print(checkCapacity(routeToPath(edgeList)))
        plt.title(title)
        print("終わり")
        print("最終経路:{}".format(penaltyFunction(edgeList, 0)))
        print("ペナルティ関数値:{}".format(penaltyFunction(edgeList, 1)))
        print("ルート数:{}".format(len(routeToPath(edgeList))))
        plt.savefig("./output/param/ini_" + filename +".png")  # save as png
        plt.show()
        return(0)


if __name__ == "__main__":
    Capa = [160, 140, 200, 200, 200, 160, 140, 200, 200, 200, 200, 200, 200, 200]
    Dura = [float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), \
            200, 160, 230, 200, 200, float("inf"), float("inf"), 720, 1040]
    Vehicle = [5, 10, 8, 12, 16, 6, 11, 9, 14, 18, 7, 10, 11, 11]

    need_D = [6, 7, 8, 9, 10, 13, 14]
    # need_D = [13, 14]
    min_m = [5, 10, 8, 12, 16, 6, 11, 9, 14, 18, 7, 10, 11, 11]
    sTime = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 0, 0, 50, 90]

    for i in range(1, 15):
        filename = "vrpnc" + str(i)
        CAPACITY = Capa[i-1]
        D = Dura[i-1]
        SERVICE = sTime[i-1]
        m = Vehicle[i-1] + 1
        if i not in need_D:
            continue

        df = createDataFrame("./csv/Christ/", filename)
        num_shelter = len(df.index)
        # num_shelter = 11
        print("顧客数:{}".format(num_shelter-1))
        print("対象ファイル名:{}".format(filename))
        print("トラック容量:{}".format(CAPACITY))
        print("ルート長制約:{}".format(D))
        print("サービス時間:{}".format(SERVICE))
        print("利用できるトラック台数+1:{}".format(m))

        # 各避難所間の移動コスト行列を生成する
        # 2次元配列costで保持
        cost = createCostMatrix(num_shelter)

        paramArray = []
        param = 0
        f = open("./output/param3/1_ini_" + filename + ".csv", "w")
        writer = csv.writer(f, lineterminator="\n")
        paramList = ["α", "値", "実行", "ルート数", "距離"]
        paramArray.append(paramList)
        paramList = []
        bestDistance  = float("inf")
        bestRoute = []

        start = time.time()
        m_time = 0
        while(True):
            print("--------------------------------------------------")
            ALPHA = 0.01 * np.power(math.pow(10, (1/10)), param)
            print("パラメータα:{}".format(ALPHA))

            # セービング法でルートを構築する
            # デポを含まない２次元配列で受け取る
            route = savingMethod(num_shelter, cost)
            # セービング方で得られた解にデポをつける
            path = createEdgeSet(route)
            path = routeSplit(path)
            print("ルート数(セービング法)：{}".format(len(path)))
            # graphPlot(pathToRoute(path), isFirst=1, isLast=0, title="Saving Route")

            random_order = [j for j in range(1, num_shelter)]
            random.shuffle(random_order)


            for n, k in enumerate(random_order):
                prePath = copy.deepcopy(path)
                local_route = Neighborhoods(k, path, f_option=1, reduce_route=1)
                path = routeToPath(local_route)
                if path == False:
                    path = copy.deepcopy(prePath)
                # graphPlot(local_route, isFirst=0, isLast=0, title="local search")
                sys.stdout.write("\r%d番目ノードの局所探索" % n)
                sys.stdout.flush()
                time.sleep(0.01)

            print("")
            print("ルート数(P関数局所探索後)：{}".format(len(path)))
            route = pathToRoute(path)
            # graphPlot(route, isFirst=1, isLast=0, title="localSearch")

            if(penaltyFunction(route, 2) > 0):
                m_start = time.time()
                route,result = modification(route)
                m_time += time.time() - m_start
                print("修正結果:{}".format(result))
                # checkCapacity(routeToPath(route))
            else:
                result = "success"
                print(result)
            # graphPlot(route, isFirst=0, isLast=0, title="modification")
            path = routeToPath(route)
            route_num = len(path)
            distance = penaltyFunction(route, 0)
            print("")
            print("ルート数(修正後)：{}".format(route_num))
            print("総距離:{}".format(distance))
            if(distance < bestDistance and result == "success"):
                bestRoute = copy.deepcopy(route)
                bestDistance = distance

            paramList = [param, ALPHA, result, route_num, distance]
            paramArray.append(paramList)
            paramList = []
            if(result == "success" and min_m[i-1] > route_num):
                print(checkCapacity(path))
                print(checkDuration(path))
                print("ペナルティ項" + str(penaltyFunction(route, 2)))


            param += 1
            if(param > 40):
                break
        elapsed_time = time.time() - start
        print("計算時間：" + str(elapsed_time) + "[sec]")
        writer.writerows(paramArray)
        writer.writerow(bestRoute)
        writer.writerow(["最短コスト:", bestDistance])
        writer.writerow(["計算時間：", elapsed_time])
        writer.writerow(["修正時間：", m_time])
        writer.writerow(["トラック容量：", CAPACITY, "期間制約", D])
        writer.writerow(["サービス時間：", SERVICE])
        f.close()
        graphPlot(route, isFirst=1, isLast=0, title="modification")
