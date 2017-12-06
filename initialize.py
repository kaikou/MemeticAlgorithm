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
# セービング値の効果をコントロールする係数
LAMBDA = 1


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
    # print(s)

    """
    経路の結合処理
    """
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

    """
    経路の出力
    """
    route = []
    heiro = []
    demand = []
    distance = []
    drt = 0
    for i in range(1, num_shelter):
        if q[i] > 0:
            ii = i
            while(True):
                heiro.append(ii)
                ii = nex[ii]
                if ii == 0:
                    distance.append(dr[i]) # その経路の移動コスト
                    demand.append(q[i]) # その経路の総需要
                    drt += dr[i]
                    route.append(heiro)
                    heiro = []
                    break

    print(route)
    print(demand)
    print(distance)
    print(drt)
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
    print(Path)
    return Path

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
        near_cost = np.append(near_cost, np.array([[i, c]]), axis=0)
    near_cost = near_cost[near_cost[:, 1].argsort()] # nodeに近い順にソート

    # print(near_cost[1:near+1, 0])
    return near_cost[1:near+1, 0]




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

    return(X, Y, N, pos, G)



"""
グラフをプロットする
"""
def graphPlot(G, N, e):
    E = []
    edge_labels = {}
    sum_cost = 0
    labels = {}

    # for i in range(num_shelter):
    #     for j in range(num_shelter):
    #         if(x[i][j] == 1):
    #             E.append((i, j))
    #             edge_labels[(i, j)] = cost[i][j]

    for edge in e:
        for i, node in enumerate(edge):
            if i == 0:
                E.append([0, node])
                edge_labels[(0, node)] = cost[0][node]
                pre_node = node
                if len(edge) == 1:
                    E.append([node, 0])
                    edge_labels[(node, 0)] = cost[node][0]
            else:
                E.append([pre_node, node])
                edge_labels[(pre_node, node)] = cost[pre_node][node]
                pre_node = node
                if i == len(edge)-1:
                    E.append([node, 0])
                    edge_labels[(node, 0)] = cost[node][0]

    for i in range(num_shelter):
        # labels[i] = df.ix[i].d
        labels[i] = i

    print(E)
    G.add_nodes_from(N)
    G.add_edges_from(E)
    nx.draw_networkx(G, pos, with_labels=False, node_color='r', node_size=200)
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 70)
    plt.ylim(0, 70)
    # plt.axis('off')
    plt.title('Delivery route')
    plt.savefig("./output/cvrp.png")  # save as png
    # plt.grid()
    plt.show()

    return(0)


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

    route = savingMethod(num_shelter, cost)
    print("ルート数：{}".format(len(route)))

    path = createEdgeSet(route)



    # lineCount = 0
    # readfile = './data/solomon_25/C101.txt'
    # for line in open(readfile, "r"):
    #     lineCount += 1
    #     if lineCount < 10:
    #         continue
    #     print(line.strip())
        # print(line)


    # X, Y, N, pos, G = createGraphList()  #グラフ描画準備
    # graphPlot(G, N, route)
