#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)

import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt



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


def createList():
    P_A = [6, 5, 8, 7, 11, 10, 1, 9, 3, 12, 4, 2]
    P_B = [2, 6, 5, 8, 12, 7, 10, 1, 9, 11, 3, 4]

    return P_A, P_B


def graphPlot(G, N):
    E = []
    edge_labels = {}
    sum_cost = 0
    labels = {}

    # for i in range(num_shelter):
    #     for j in range(num_shelter):
    #         if(x[i][j].value() == 1.0):
    #             sum_cost += cost[i][j]
    #             print(i, j, x[i][j].value())
    #             E.append((i, j))
    #             edge_labels[(i, j)] = cost[i][j]
    #
    # print("総移動コスト" + str(sum_cost))

    for i in range(num_shelter):
        # labels[i] = df.ix[i].d
        labels[i] = i


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
    plt.grid()
    plt.show()

    return(0)





if __name__ == '__main__':

    df = createDataFrame("./data/", "data_r101")
    num_shelter = 11

    P_A, P_B = createList()
    print("P_A:" + str(P_A))
    print("P_B:" + str(P_B))

    X, Y, N, pos, G = createGraphList()  #グラフ描画準備

    cost = createCostMatrix(num_shelter)

    graphPlot(G, N)
