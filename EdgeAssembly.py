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
    # print("コスト行列-----------------------------------")
    # print(arr)
    # print("---------------------------------------------")
    np.savetxt("./output/cost.csv", arr, delimiter=',', fmt='%.2f')
    return arr

"""
親Aと親Bの遺伝子を作るだけの関数
@INPUT:
    None
@OUTPUT:
    P_A : 親Aの遺伝子情報
    P_B : 親Bの遺伝子情報
"""
def createList():
    P_A = [6, 5, 8, 7, 11, 10, 1, 9, 3, 12, 4, 2]
    P_B = [2, 6, 5, 8, 12, 7, 10, 1, 9, 11, 3, 4]

    # P_A = [6, 5, 8, 7, 11, 10, 1, 9, 12, 3, 4, 2]
    # P_B = [6, 5, 8, 7, 10, 11, 12, 1, 9, 3, 4, 2]

    return P_A, P_B

def createEdge(Parent):
    # 配送順序の配列を変数genomにコピー
    genom = Parent
    total_cost = 0
    route_flag = False
    # どの避難所間を通ったかを示す2次元配列を0で初期化
    x = np.zeros((num_shelter, num_shelter), int) #小数点以下を加える→float型
    for i in range(len(genom)):
        # ルート区切り番号だった場合
        if genom[i] > num_shelter - 1: # >10
            if route_flag == True:
                x[genom[i-1]][0] = 1
                print("{}→{}".format(genom[i-1], 0))
            route_flag = False
            print("_{}_区切り".format(genom[i]))
        else : # ルート区切り番号ではない場合(避難所番号)

            # 現在参照している避難所番号の前が区切り番号だった，
            # もしくは遺伝子の最初を参照している場合
            if route_flag == False:
                x[0][genom[i]] = 1
                route_flag = True
                print("{}→{}".format(0, genom[i]))
            else : # フラグがTrue，つまり経路続行
                x[genom[i-1]][genom[i]] = 1
                print("{}→{}".format(genom[i-1], genom[i]))
    # 遺伝子の最後の番号が区切り番号でない場合，
    if route_flag == True:
        x[genom[i]][0] = 1
        print("{}→{}".format(genom[i], 0))

    #総移動コストの計算
    for i in range(num_shelter):
        for j in range(num_shelter):
            total_cost += cost[i][j] * x[i][j]
    # 総移動コストと，移動エッジ行列を返す
    return total_cost, x


def EAX(x_A, x_B):
    x_AB = []
    E_A = []
    E_B = []
    AB_cycle = []

    for i in range(num_shelter):
        for j in range(i, num_shelter):
            if(x_A[i][j] == 1 or x_A[j][i] == 1):
                E_A.append([i, j])
            if(x_B[i][j] == 1 or x_B[j][i] == 1):
                E_B.append([i, j])
    # エッジの向きまで考慮
    # for i in range(num_shelter):
    #     for j in range(num_shelter):
    #         if(x_A[i][j] == 1):
    #             E_A.append([i, j])
    #         if(x_B[i][j] == 1):
    #             E_B.append([i, j])
    print(E_A)
    print(E_B)
    # G_ABを作成
    G_AB = sorted([x for x in E_A + E_B if not (x in E_A and x in E_B)])
    print("G_AB:{}".format(G_AB))

    # これいる？
    # x_AB = np.zeros((num_shelter, num_shelter), int) #小数点以下を加える→float型
    # for i in range(num_shelter):
    #     for j in range(i, num_shelter):
    #         for k, l in G_AB:
    #             x_AB[k][l] = 1

    # print(E_A)
    # print(E_B)
    # return G_AB, edgelist

    """
    AB-cycleの処理
    """
    i = 0
    s = 0
    R_A = sorted([x for x in G_AB if (x and x in E_A)])
    R_B = sorted([x for x in G_AB if (x and x in E_B)])
    P = [0]
    C = []
    print("R_A:{}".format(R_A))
    print("R_B:{}".format(R_B))
    print("R_Aの長さ:{}".format(len(R_A)))
    print("R_Aの長さ:{}".format(len(R_B)))


    while(len(G_AB)):
        ABflag = False
        v_e = random.choice(np.unique(R_A))
        print("v_e:{}".format(v_e))
        v_e_1 = v_e # 最初の端点を保持する
        while (not(ABflag)):
            if(P[s] in R_B or s == 0):
                R_A = sorted([x for x in R_A if (x and x in G_AB)])
                # 上で選択したノードv_eにつながるR_Aのエッジをeにセットする
                e = random.choice(list(filter(lambda x: v_e in x, R_A)))
                print("e:{}".format(e))
                G_AB = [x for x in G_AB if x != e]
                print("G_AB:{}".format(G_AB))
            else:
                R_B = sorted([x for x in R_B if (x and x in G_AB)])
                # 上で選択したノードv_eにつながるR_Bのエッジをeにセットする
                e = random.choice(list(filter(lambda x: v_e in x, R_B)))
                print("e:{}".format(e))
                G_AB = [x for x in G_AB if x != e]
                print("G_AB:{}".format(G_AB))
            # eの端点のv_eでない方を新たにv_eとする
            v_e = e[0] if v_e == e[1] else e[1]
            print("次の端点:{}".format(v_e))
            s += 1
            P.append(e)
            # print("s:{}".format(s))
            # print("P[s]:{}".format(P))

            # PがAB-cycleを含んでいるか判定
            if(isRoute(P[1:], v_e_1, v_e)):
                C.append(P[1:])
                P = [0]
                s = 0
                R_A = sorted([x for x in R_A if (x and x in G_AB)])
                ABflag = True
                print("C:{}".format(C))
                # if(len(R_A) % 2 == 1 or len(R_B) % 2 == 1):
                #     break

    for i, x in enumerate(C):
        print("C{}:{}".format(i, x))


def isRoute(edgeList, v_e_1, v_e):
    # エッジリストの長さが偶数かどうか
    if len(edgeList) % 2 == 0:
        if v_e_1 == v_e:
            return 1
    else:
        return 0




"""
グラフをプロットする
"""
def graphPlot(G, N, x):
    E = []
    edge_labels = {}
    sum_cost = 0
    labels = {}

    for i in range(num_shelter):
        for j in range(num_shelter):
            if(x[i][j] == 1):
                E.append((i, j))
                edge_labels[(i, j)] = cost[i][j]

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

    cost = createCostMatrix(num_shelter)

    P_A, P_B = createList()
    print("P_A:" + str(P_A))
    print("P_B:" + str(P_B))

    total_cost_A, x_A = createEdge(P_A)
    total_cost_B, x_B = createEdge(P_B)

    # print("x_A")
    # print(x_A)
    # print("x_B")
    # print(x_B)
    print("Aの総移動コスト:{}".format(total_cost_A))
    print("Bの総移動コスト:{}".format(total_cost_B))

    # G_AB, edgelist = EAX(x_A, x_B)
    EAX(x_A, x_B)
    # print(G_AB)
    # print(edgelist)

    # X, Y, N, pos, G = createGraphList()  #グラフ描画準備
    # graphPlot(G, N, G_AB)
    # X, Y, N, pos, G = createGraphList()
    # graphPlot(G, N, x_A)
    # X, Y, N, pos, G = createGraphList()
    # graphPlot(G, N, x_B)
