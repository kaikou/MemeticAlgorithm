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
CAPACITY = 80
# セービング値の効果をコントロールする係数
LAMBDA = 1
# N_near()関数で，どこまで近くのノードに局所探索するか
NEAR = 10
# penaltyFunction()で，容量制約違反に課すペナルティの係数
ALPHA = 2
# penaltyFunction()で，経路長違反に課すペナルティの係数
BETA = 1.0
# penaltyFunction()で，経路長違反とする距離
D = 100


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
        for j in range(i+1, num_shelter):
            s[i][j] = cost[i][0] + cost[0][j] - (LAMBDA * cost[i][j])
    # print(s) #セービング値

    """
    経路の結合処理
    """
    random_order = [i for i in range(1, num_shelter)]
    random.shuffle(random_order)

    smax = 1
    while(smax != 0):
        smax = 0
        for i in range(1, num_shelter): #1から順に
        # for i in random_order:
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

    """
    経路の出力
    """
    route = []
    heiro = []
    demand = []
    distance = []
    # plot = []
    drt = 0
    for i in range(1, num_shelter):
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

    print(route)
    print(demand)
    print(distance)
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
２次元配列を生成する．
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
    path = routeToPath(route)
    # ルート総距離
    for e in route:
        F += cost[e[0]][e[1]]

    if option == 0:
        return F

    for edges in path:

        """
        容量制約違反の計算
        """
        nodes = np.unique(edges)
        for n in nodes: # 各ルートの合計需要
            R_demands += df.ix[n].d

        if R_demands > CAPACITY:
            F_c += R_demands - CAPACITY # ルート内の需要超過
        else:
            F_c += 0
            # F_c += abs(R_demands - CAPACITY) # ルート内の需要超過
        R_demands = 0

        """
        route duration制約違反の計算
        """
        # for e in edges:
        #     R_cost += cost[e[0]][e[1]]
        # if R_cost > D:
        #     F_d += R_cost - D
        # else:
        #     F_d += 0
        #     # F_d += abs(R_cost - D)
        # R_cost = 0

    # ペナルティ関数

    # F_p = F + (ALPHA * F_c) + (BETA * F_d)
    F_p = F + (ALPHA * F_c)
    # print("F:{}, F_c:{}, F_d:{}".format(F, F_c, F_d))
    return F_p


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
    for i, c in enumerate(cost[node][:]):
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
        1: ペナルティ関数による評価
        0: 総移動コストのみの評価
"""
def Neighborhoods(v, path, neighbor, f_option):
    EdgeSet = []
    DefaultEdgeSet = []
    path_cost = 0

    # 解を構成する全てのエッジ集合を生成
    for edge in path:
        for j in edge:
            EdgeSet.append(j)
    DefaultEdgeSet = copy.deepcopy(EdgeSet)

    # 元の解のペナルティ関数評価値を保持
    P_eval = penaltyFunction(DefaultEdgeSet, f_option)
    # print("デフォルト:{}".format(DefaultEdgeSet))

    # 渡されたノードvに繋がるエッジ2つ
    link_v = [i for i in EdgeSet if (v in i)]
    v_minus = link_v[0][0] if v == link_v[0][1] else link_v[0][1]
    v_plus = link_v[1][1] if v == link_v[1][0] else link_v[1][0]
    # print("link_v:{}".format(link_v))
    # print("link_v[0]:{}".format(link_v[0])) # ノードvに向かうエッジ
    # print("link_v[1]:{}".format(link_v[1])) # ノードvを出るエッジ

    # ノードvからnearだけ近いノードをそれぞれwとして選ぶ
    for w in N_near(v, NEAR):
        link_w = [i for i in EdgeSet if (w in i)]

        if(len(link_v) < 2 or len(link_w) < 2):
            # print("ノードに対するエッジが2つない")
            EdgeSet = copy.deepcopy(DefaultEdgeSet)
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
        if neighbor == "10inter":
            # print("(1,0)Interchange①やるよー")
            """
            (1,0)Interchange①
            """
            # 選んだwに対して
            # 元々繋がっているエッジ
            l1 = cost[w_minus][w] # w-→w
            l2 = cost[v_minus][v] # v-→v
            l3 = cost[v][v_plus] # v→v+
            # つなぎ直すエッジ
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
            except ValueError:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue

            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue

            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue
            # ペナルティ関数により評価
            if penaltyFunction(EdgeSet, f_option) > P_eval:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
            else:
                # print("{}適用".format(neighbor))
                return EdgeSet

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
            except ValueError:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue

            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue

            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue
            # ペナルティ関数により評価
            if penaltyFunction(EdgeSet, f_option) > P_eval:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
            else:
                # print("{}適用".format(neighbor))
                return EdgeSet

            return EdgeSet

        """
        (0,1)Interchange
        """
        if neighbor == "01inter":
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
            except ValueError:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue

            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue

            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue
            # ペナルティ関数により評価
            if penaltyFunction(EdgeSet, f_option) > P_eval:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
            else:
                # print("{}適用".format(neighbor))
                return EdgeSet

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
            except ValueError:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue
            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue

            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue
            # ペナルティ関数により評価
            if penaltyFunction(EdgeSet, f_option) > P_eval:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
            else:
                # print("{}適用".format(neighbor))
                return EdgeSet

            return EdgeSet

        """
        (1,1)Interchange
        """
        if neighbor == "11inter":
            # print("(1,1)Interchangeやるよー")

            """
            (1,1)Interchange①
            """

            """
            w--を求めるステップ
            """
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

            try:
                # リストからノードw-でない方をノードw--として選択
                w_minus_minus = w_minus2list[0][0] if w_minus2list[0][1] == w_minus \
                else w_minus2list[0][1]
            except IndexError:
                continue

            # 同じエッジを選ぶ可能性があるものは排除
            # if link_v[0] == w_minus2list[0] or link_v[1] == w_minus2list[0] \
            # or link_w[0] == w_minus2list[0] or link_w[1] == w_minus2list[0]:
            #     continue

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
            except ValueError:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue

            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue

            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                # print("リセット1")
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue
            # ペナルティ関数により評価
            if penaltyFunction(EdgeSet, f_option) > P_eval:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
            else:
                # print("{}適用".format(neighbor))
                return EdgeSet

            """
            (1,1)Interchange②
            """

            """
            w++を求めるステップ
            """
            # ノードw+を持つリストを作る
            w_plus2list = [x for x in path[route_num] \
                            if(w_plus in x and x != link_w[1])]

            try:
                # リストからノードw+でない方をノードw++として選択
                w_plus_plus = w_plus2list[0][1] if w_plus2list[0][0] == w_plus \
                else w_plus2list[0][0]
            except IndexError:
                continue

            # print("{}を含むルートは:{}".format(w, path[route_num]))
            # print("リスト:{}".format(w_plus2list[0]))
            # print("w+:{}".format(w_plus))
            # print("w++:{}".format(w_plus_plus))

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
            except ValueError:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue

            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue

            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                # print("リセット２")
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue

            # ペナルティ関数により評価
            if penaltyFunction(EdgeSet, f_option) > P_eval:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
            else:
                # print("{}適用".format(neighbor))
                return EdgeSet

            return EdgeSet

        """
        2-opt近傍
        """
        if neighbor == "2opt":
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
            except ValueError:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue

            for es in EdgeSet:
                if es[0] == es[1]:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue


            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue

            # ペナルティ関数により評価
            if penaltyFunction(EdgeSet, f_option) > P_eval:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
            else:
                # print("{}適用".format(neighbor))
                return EdgeSet

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
            except ValueError:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue

            for es in EdgeSet:
                if es[0] == es[1]:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue

            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                continue

            # ペナルティ関数により評価
            if penaltyFunction(EdgeSet, f_option) > P_eval:
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
            else:
                # print("{}適用".format(neighbor))
                return EdgeSet

            # vとwが同じルートか判定
            sameRoute = False
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
                except ValueError:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue

                for es in EdgeSet:
                    if es[0] == es[1]:
                        EdgeSet = copy.deepcopy(DefaultEdgeSet)
                        continue

                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue

                # ペナルティ関数により評価
                if penaltyFunction(EdgeSet, f_option) > P_eval:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                else:
                    # print("{}適用".format(neighbor))
                    return EdgeSet

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
                except ValueError:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1]:
                        EdgeSet = copy.deepcopy(DefaultEdgeSet)
                        continue
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    continue

                # ペナルティ関数により評価
                if penaltyFunction(EdgeSet, f_option) > P_eval:
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                else:
                    # print("{}適用".format(neighbor))
                    return EdgeSet
                return EdgeSet
    return EdgeSet


"""
2次元のエッジリストから，各閉路毎にエッジを持つ3次元リストに変換する
@INPUT:
    route: エッジの２次元リスト
@OUTPUT:
    Path: ルート情報を含むエッジの3次元リスト
    False: 順回路を構築できない
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
        if e[0] == 0 and e[1] == 0:
            route.remove(e)
            continue
        # エッジ端のどちらかに0を含むか
        elif e[0] == 0 or e[1] == 0:
            R.append(e)
            route.remove(e)
            # 0じゃない方をv_eにセット
            v_e = e[1] if e[0] == 0 else e[0]
        else: # 0を含まない閉路を発見
            R.append(e)
            route.remove(e)
            v_e = e[1]
            v_e_1 = e[0]
            heiro = True

        while(not(find_flag)):
            # v_eを含むエッジをroute内から探しeにセット
            if (v_e not in np.unique(route)):
                print("見つからない")
                return False
            e = random.choice(list(filter(lambda x: v_e in x, route)))
            R.append(e)
            route.remove(e)
            # print("-------------------")
            # print("v_e:{}".format(v_e))
            # print("v_eを含むエッジ:{}".format(e))
            # print("route:{}".format(route))
            # print("Path:{}".format(Path))

            # eの端点のv_eでない方を新たにv_eとする
            v_e = e[0] if v_e == e[1] else e[1]
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
    I: 部分順回路のインデックスリスト
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
グラフのリストを作成する
@INPUT:
    None
@OUTPUT:
    X:
    Y:
    N:
    pos:
    G:
"""
def createGraphList():
    X = []
    Y = []
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

    return(N, pos, G)

"""
グラフをプロットする
"""
def graphPlot(edgeList, isFirst, isLast):
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
        edge_labels[(e[0], e[1])] = cost[e[0]][e[1]]

    for i in range(num_shelter):
        # labels[i] = df.ix[i].d
        labels[i] = i

    G.add_nodes_from(N)
    G.add_edges_from(E)
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color="r")
    nx.draw_networkx_edges(G, pos, width=1)
    # nx.draw_networkx(G, pos, with_labels=False, node_color='r', node_size=80) # デフォルト200
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=4) # デフォルト12
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=3) # デフォルト8

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 70)
    plt.ylim(0, 70)
    # plt.axis('off')
    plt.title('Delivery route')
    # plt.grid()

    # 元の経路
    if isFirst == 1:
        print("最初の経路:{}".format(penaltyFunction(edgeList, 0)))
        print("ペナルティ関数値:{}".format(penaltyFunction(edgeList, 1)))
        print("ルート数:{}".format(len(routeToPath(edgeList))))
        plt.title('Initial Delivery route')
        plt.pause(0.01)
        plt.figure()

    # 連続プロット中
    if isLast == 0:
        plt.pause(0.01)
        plt.clf()
    else:
        print("終わり")
        print("最終経路:{}".format(penaltyFunction(edgeList, 0)))
        print("ペナルティ関数値:{}".format(penaltyFunction(edgeList, 1)))
        print("ルート数:{}".format(len(routeToPath(edgeList))))
        plt.savefig("./output/" + filename +".png")  # save as png
        plt.show()
        return(0)


if __name__ == "__main__":
    filename = "R101"

    df = createDataFrame("./csv/", filename)
    num_shelter = len(df.index)
    num_shelter = 31

    # 各避難所間の移動コスト行列を生成する
    # 2次元配列costで保持
    cost = createCostMatrix(num_shelter)

    print(df[:11])
    # print(cost)

    start = time.time()
    # セービング法でルートを構築する
    # デポを含まない２次元配列で受け取る
    route = savingMethod(num_shelter, cost)
    elapsed_time = time.time() - start
    print("計算時間：" + str(elapsed_time) + "[sec]")
    print("ルート数：{}".format(len(route)))

    # セービング方で得られた解にデポをつける
    path = createEdgeSet(route)
    # localSearch(path)

    # test = [[0,1], [2, 1], [2, 3], [3, 0], [4, 5], [5, 6], [4, 0], [6, 0], \
    #         [7, 8], [8, 9], [7, 9], [10, 11], [12, 11], [12, 10]]
    # path = routeToPath(test)
    # print(isHeiro(path))

    random_order = [i for i in range(1, num_shelter)]
    random.shuffle(random_order)

    graphPlot(pathToRoute(path), isFirst=1, isLast=0)
    print(route)
    for n, i in enumerate(random_order):
        prePath = copy.deepcopy(path)
        local_route = Neighborhoods(i, path, "10inter", 1)
        path = routeToPath(local_route)
        local_route = Neighborhoods(i, path, "11inter", 1)
        path = routeToPath(local_route)
        local_route = Neighborhoods(i, path, "01inter", 1)
        path = routeToPath(local_route)
        local_route = Neighborhoods(i, path, "2opt", 1)
        path = routeToPath(local_route)

        graphPlot(local_route, isFirst=0, isLast=0)

        if path == False:
            path = copy.deepcopy(prePath)
        # print("{}回目".format(n))
        sys.stdout.write("\r%d個目" % n)
        sys.stdout.flush()
        time.sleep(0.01)
    route = pathToRoute(path)

    print(path)
    graphPlot(route, isFirst=0, isLast=1)
