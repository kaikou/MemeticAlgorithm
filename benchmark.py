#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)
#

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys


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
    print(df[0:11])
    print("コスト行列-----------------------------------")
    print(arr)
    print("---------------------------------------------")
    np.savetxt("./output/cost.csv", arr, delimiter=',', fmt='%.2f')
    return arr


"""
グラフをプロットする
"""
def graphPlot(edgeList, depot, title):
    N = []
    G = nx.Graph()
    pos = {}  #ノードの位置情報格納
    
    if depot == 0:
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
        # G.add_edges_from(E)
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color="r")
        nx.draw_networkx_edges(G, pos, width=1)
        # nx.draw_networkx(G, pos, with_labels=False, node_color='r', node_size=80) # デフォルト200
        # nx.draw_networkx_labels(G, pos, labels=labels, font_size=6) # デフォルト12
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6) # デフォルト8

        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        # plt.xlim(0, 70)
        # plt.ylim(0, 70)
        # plt.axis('off')
        # plt.grid()

    # 連続プロット中
        plt.title(title)
        # plt.savefig("./output/problem/" + title)  # save as png
        plt.pause(0.01)
        # plt.clf()
    else:
        N.append(0)
        pos[0] = (df.ix[0].x, df.ix[0].y)

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
        # G.add_edges_from(E)
        nx.draw_networkx_nodes(G, pos, node_size=40, node_color="b")
        nx.draw_networkx_edges(G, pos, width=1)
        # nx.draw_networkx(G, pos, with_labels=False, node_color='r', node_size=80) # デフォルト200
        # nx.draw_networkx_labels(G, pos, labels=labels, font_size=6) # デフォルト12
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6) # デフォルト8

        plt.legend()
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")

        plt.title(title)
        plt.savefig("./output/problem/" + title)  # save as png
        plt.pause(0.01)
        plt.clf()

if __name__ == "__main__":
    name = "vrpnc"
    # name = ["75a", "75b", "75c", "75d", "100a", "100b", "100c", "100d", "150a", "150b", "150c", "150d", "385"]
    filename = ""
    for i in range(1, 15):
        filename = name + str(i)
        # filename.append("tai" + i)
        print(filename)

    # for i in filename:
        df = createDataFrame("./csv/Christ/", filename)
        num_shelter = len(df.index)
        # num_shelter = 11
        print("顧客数:{}".format(num_shelter-1))

        # 各避難所間の移動コスト行列を生成する
        # 2次元配列costで保持
        cost = createCostMatrix(num_shelter)
        graphPlot("", 0, "Problem" + str(i))
        graphPlot("", 1, "Problem" + str(i))
