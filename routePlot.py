#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)


import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys, time
import copy
import itertools


# 使用できる車両数
m = 5
# 車両の最大積載量
CAPACITY = 160
# セービング値の効果をコントロールする係数
LAMBDA = 1
# N_near()関数で，どこまで近くのノードに局所探索するか
NEAR = 10
# penaltyFunction()で，容量制約違反に課すペナルティの係数
ALPHA = 0.3
# penaltyFunction()で，経路長違反に課すペナルティの係数
BETA = 1.0
# penaltyFunction()で，経路長違反とする距離
D = float("inf")

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
ペナルティ関数による評価を行う
@INPUT:
    route: 解の２次元リスト
    option: どのように評価するか
        2: ペナルティ項のみ
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
        F += cost[int(e[0])][int(e[1])]

    if option == 0:
        return round(F, 2)

    for edges in path:
        # 一つの巡回路について
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
        for e in edges:
            R_cost += cost[int(e[0])][int(e[1])]
        if R_cost > D:
            F_d += R_cost - D
        else:
            F_d += 0
            # F_d += abs(R_cost - D)
        R_cost = 0

    # ペナルティ関数

    F_p = F + (ALPHA * F_c) + (BETA * F_d)
    # F_p = F + (ALPHA * F_c)
    # print("F:{}, F_c:{}, F_d:{}".format(F, F_c, F_d))
    if option == 1:
        return round(F_p, 2) # ペナルティ間数値
    elif option == 2:
        # print("ペナルティ項の合計:{}".format(round(F_c + F_d, 2)))
        return round(F_c + F_d, 2) # ペナルティ項のみ



"""
2次元のエッジリストから，各閉路毎にエッジを持つ3次元リストに変換する
@INPUT:
    route: エッジの２次元リスト
@OUTPUT:
    Path: ルート情報を含むエッジの3次元リスト
"""
def routeToPath(route):
    route.sort(key=lambda x:x[0])
    route.sort(key=lambda x:x[1])

    Path = []
    R = []
    v_e_1 = 0
    print(route)

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
                print("【routeToPath】見つからない")
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
modification()中に用いる
入力された3次元リストを元に，トラックの容量超過をしているルートの
インデックスとルート毎の総需要量のリストを返す
"""
def checkCapacity(path):
    excess = []
    excess_0 = []
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
    # print("超過ルートインデックス:{}".format(r))
    # print(excess)
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


def plotDepot(title):
    N = []
    G = nx.Graph()
    pos = {}  #ノードの位置情報格納

    N.append(0)
    pos[0] = (df.ix[0].x, df.ix[0].y)

    E = []
    edge_labels = {}
    sum_cost = 0
    labels = {}
    # for e in edgeList:
    #     E.append(e)
    #     edge_labels[(int(e[0]), int(e[1]))] = int(cost[int(e[0])][int(e[1])])

    for i in range(num_shelter):
        # labels[i] = df.ix[i].d
        labels[i] = i

    G.add_nodes_from(N)
    # G.add_edges_from(E)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="b")
    nx.draw_networkx_edges(G, pos, width=1)
    # nx.draw_networkx(G, pos, with_labels=False, node_color='r', node_size=80) # デフォルト200
    # nx.draw_networkx_labels(G, pos, labels=labels, font_size=6) # デフォルト12
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6) # デフォルト8

    plt.legend()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title(title)
    plt.savefig("./output/test/" + filename +".png")  # save as png
    plt.show()
    return(0)

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
    # nx.draw_networkx_labels(G, pos, labels=labels, font_size=6) # デフォルト12
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6) # デフォルト8

    plt.legend()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    # plt.xlim(0, 70)
    # plt.ylim(0, 70)
    # plt.axis('off')
    # plt.grid()

    plot_path = routeToPath(edgeList)

    # 元の経路
    if isFirst == 1:
        print("最初の経路:{}".format(penaltyFunction(edgeList, 0)))
        print("ペナルティ関数値:{}".format(penaltyFunction(edgeList, 1)))
        print("ルート数:{}".format(len(plot_path)))
        plt.title(title)
        plt.pause(0.01)
        plt.figure()

    # 連続プロット中
    if isLast == 0:
        plt.title(title)
        plt.pause(0.01)
        plt.clf()
    else:
        path = routeToPath(edgeList)
        print(checkCapacity(path))
        # print(checkDuration(path))
        plt.title(title)
        print("終わり")
        print("最終経路:{}".format(penaltyFunction(edgeList, 0)))
        print("ペナルティ関数値:{}".format(penaltyFunction(edgeList, 1)))
        print("ルート数:{}".format(len(plot_path)))
        plotDepot(title)

if __name__ == '__main__':

    filename = "vrpnc1"

    # 避難所情報のデータフレームを生成する
    # 引数[0]:ファイルパス，[1]:ファイル名
    df = createDataFrame("./csv/Christ/", filename)
    num_shelter = len(df.index)
    # num_shelter = 21
    result_df = pd.DataFrame(index=[], columns=['世代', '総移動コスト'])

    # 各避難所間の移動コスト行列を生成する
    # 2次元配列costで保持
    cost = createCostMatrix(num_shelter)


    # route = [[0, 49], [49, 10], [10, 9], [9, 50], [50, 29], [29, 21], [21, 34], [34, 30], [30, 39], [39, 33], [33, 45], [45, 15], [15, 0], [0, 47], [47, 4], [4, 18], [18, 0], [0, 27], [27, 48], [48, 8], [8, 23], [23, 6], [6, 0], [0, 14], [14, 25], [25, 13], [13, 41], [41, 40], [40, 19], [19, 42], [42, 44], [44, 37], [37, 17], [17, 0], [0, 32], [32, 11], [11, 2], [2, 16], [16, 0], [0, 38], [38, 5], [5, 12], [12, 46], [46, 0], [0, 1], [1, 22], [22, 20], [20, 35], [35, 36], [36, 3], [3, 0], [0, 28], [28, 31], [31, 26], [26, 7], [7, 43], [43, 24], [24, 0]]

    path = [[[0, 6], [27, 6], [27, 0]], [[0, 17], [17, 37], [37, 44], [44, 42], [42, 19], [19, 40], [40, 41], [41, 13], [13, 25], [25, 14], [14, 0]], [[0, 20], [35, 20], [36, 35], [3, 36], [22, 3], [1, 22], [1, 32], [0, 32]], [[0, 24], [43, 24], [7, 43], [23, 7], [48, 23], [8, 48], [8, 26], [26, 31], [31, 28], [28, 0]], [[0, 34], [34, 30], [30, 39], [39, 33], [33, 45], [45, 15], [15, 0]], [[2, 16], [38, 16], [5, 38], [12, 5], [12, 0], [10, 0], [49, 10], [9, 49], [9, 50], [50, 29], [29, 21], [21, 0], [47, 0], [4, 47], [18, 4], [18, 0], [46, 0], [46, 11], [11, 2]]]

    # for r in path:
    #     print(r)



    print(len(path))
    route = pathToRoute(path)

    apath = routeToPath(route)
    print(len(apath))

    # graphPlot(route, isFirst=0, isLast=1, title=filename + " testPlot")
