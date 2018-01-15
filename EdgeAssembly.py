#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)

import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import sys, time
import copy
import itertools

# 車両の最大積載量
CAPACITY = 60

# penaltyFunction()で，容量制約違反に課すペナルティの係数
ALPHA = 5
# penaltyFunction()で，経路長違反に課すペナルティの係数
BETA = 1.0
# penaltyFunction()で，経路長違反とする距離
D = 40


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


"""
車両が通るエッジを表す行列を生成
@INPUT：
    ga : エッジを生成するgenomClass
@OUTPUT:
    E : 個体のエッジ集合
    total_cost：個体の移動コスト
"""
def createEdgeSet(genom):
    # 配送順序の配列を変数genomにコピー
    # genom = ga.getGenom()
    total_cost = 0
    route_flag = False
    E = []

    for i in range(len(genom)):
        # ルート区切り番号だった場合
        if genom[i] > num_shelter - 1: # >10
            if route_flag == True:
                E.append([genom[i-1], 0])
            route_flag = False
        else : # ルート区切り番号ではない場合(避難所番号)

            # 現在参照している避難所番号の前が区切り番号だった，
            # もしくは遺伝子の最初を参照している場合
            if route_flag == False:
                E.append([0, genom[i]])
                route_flag = True
            else : # フラグがTrue，つまり経路続行
                E.append([genom[i-1], genom[i]])
    # 遺伝子の最後の番号が区切り番号でない場合，
    if route_flag == True:
        E.append([genom[i], 0])

    #総移動コストの計算
    for e in E:
        total_cost += cost[e[0]][e[1]]

    # 移動エッジ行列と，総移動コストを返す
    return E, total_cost

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
        nodes = np.unique(edges)
        for n in nodes: # 各ルートの合計需要
            R_demands += df.ix[n].d
        F_c += abs(R_demands - CAPACITY) # ルート内の需要超過
        R_demands = 0

        for e in edges:
            R_cost += cost[e[0]][e[1]]
        if R_cost <= D:
            R_cost = 0
            # abs(R_cost)
        F_d += R_cost - D
        R_cost = 0

    # ペナルティ関数
    F_p = F + (ALPHA * F_c) + (BETA * F_d)
    return F_p


def EAX(E_A, E_B):
    x_AB = []
    AB_cycle = []

    # E_A = [[0, 1], [1, 3], [3, 9], [9, 20], [20, 0], [0, 2], [2, 15], [15, 0], [0, 4], [4, 12], [12, 13], [13, 0], [0, 5], [5, 14], [14, 16], [16, 17], [17, 18], [18, 0], [0, 6], [6, 7], [7, 8], [8, 10], [10, 11], [11, 19], [19, 0]]
    # E_B = [[0, 1], [1, 3], [3, 9], [9, 20], [20, 0], [0, 2], [2, 4], [4, 15], [15, 0], [0, 5], [5, 14], [14, 16], [16, 17], [17, 18], [18, 0], [0, 6], [6, 7], [7, 8], [8, 10], [10, 11], [11, 19], [19, 0], [0, 12], [12, 13], [13, 0]]

    print(E_A)
    print(E_B)
    # graphPlot(E_A, isFirst=1, isLast=0, title="E_A")
    # graphPlot(E_B, isFirst=1, isLast=0, title="E_B")
    # G_ABを作成
    G_AB = [x for x in E_A + E_B if not (x in E_A and x in E_B)]

    # G_ABから[0, 2]と[2, 0]のような同じエッジを示す要素を排除
    for e1, e2 in itertools.combinations(G_AB, 2):
        if e1[0] == e2[1] and e1[1] == e2[0]:
            G_AB.remove(e1)
            G_AB.remove(e2)

    print("G_AB:{}".format(G_AB))
    # graphPlot(G_AB, isFirst=1, isLast=0, title="G_AB")

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
    R_A = [x for x in G_AB if (x and x in E_A)]
    R_B = [x for x in G_AB if (x and x in E_B)]
    P = [0]
    C = []
    print("R_A:{}".format(R_A))
    print("R_B:{}".format(R_B))
    print("R_Aの長さ:{}".format(len(R_A)))
    print("R_Bの長さ:{}".format(len(R_B)))


    while(len(G_AB)):
        ABflag = False
        v_e = random.choice(np.unique(R_A))
        # print("v_e:{}".format(v_e))
        v_e_1 = v_e # 最初の端点を保持する
        while (not(ABflag)):
            if(P[s] in R_B or s == 0):
                R_A = [x for x in R_A if (x and x in G_AB)]
                # 上で選択したノードv_eにつながるR_Aのエッジをeにセットする
                e = random.choice(list(filter(lambda x: v_e in x, R_A)))
                # print("e:{}".format(e))
                G_AB = [x for x in G_AB if x != e]
                # print("G_AB:{}".format(G_AB))
            else:
                R_B = [x for x in R_B if (x and x in G_AB)]
                # 上で選択したノードv_eにつながるR_Bのエッジをeにセットする
                e = random.choice(list(filter(lambda x: v_e in x, R_B)))
                # print("e:{}".format(e))
                G_AB = [x for x in G_AB if x != e]
                # print("G_AB:{}".format(G_AB))
            # eの端点のv_eでない方を新たにv_eとする
            v_e = e[0] if v_e == e[1] else e[1]
            # print("次の端点:{}".format(v_e))
            s += 1
            P.append(e)
            # print("s:{}".format(s))
            # print("P[s]:{}".format(P))

            # PがAB-cycleを含んでいるか判定
            if(isRoute(P[1:], v_e_1, v_e)):
                C.append(P[1:])
                P = [0]
                s = 0
                R_A = [x for x in R_A if (x and x in G_AB)]
                ABflag = True


    for i, x in enumerate(C):
        print("C{}:{}".format(i, x))


    """
    ステップ3:E-setを構成する
    """
    E_set = random.choice(C) # Single戦略
    print("E-set:{}".format(E_set))
    # graphPlot(E_set, isFirst=1, isLast=0, title="E-set")

    """
    ステップ4:E-setを用いて中間個体を生成する
    """
    # E_AからE-setに含まれるE_Aに属する枝を取り除く
    interA = [x for x in E_A if not(x and x in E_set)]
    # E-setに含まれるE_Bに属する枝を付け加える
    interB = [x for x in E_B if (x and x in E_set)]
    intermediate = interA + interB

    print("中間:{}".format(intermediate))

    """
    ステップ5:部分順回路が含まれる場合，結合する
    """

    # intermediate = [[6, 5], [5, 8], [7, 1], [10, 1], [0, 9], [3, 0], [0, 4], \
    #  [2, 0], [2, 6], [8, 0], [7, 10], [9, 0], [3, 4]]

    # intermediate = [[6, 5], [5, 8], [7, 1], [10, 1], [0, 9], [3, 0], [0, 4], \
    #  [2, 0], [0, 2], [8, 6], [7, 10], [9, 0], [3, 4]]

    print(isHeiro(routeToPath(intermediate)))
    subtour = isHeiro(routeToPath(intermediate))
    if(subtour != 0):
        child = EAXstep5(intermediate, subtour)
    else:
        child = intermediate

    return child


"""
EAXのステップ5を処理する
中間個体に部分巡回路が含まれる場合に，部分巡回路をランダムな順番で選択し，
m個のルートのどれかに結合することでm個のルートからなる子を得る．
@INPUT:
    intermediate：中間個体のエッジ集合
    subtourIndex：部分順回路のルートインデックス
@OUTPUT:
    child：子個体
"""
def EAXstep5(intermediate, subtourIndex):
    while(subtourIndex != 0):
        best = 0
        for e in intermediate:
            best += cost[e[0]][e[1]]

        Ui = routeToPath(intermediate)
        subnum = random.choice(subtourIndex)
        Ur = Ui[subnum]
        Ui.pop(subnum)

        print("Ui:{}".format(Ui))
        print("Ur:{}".format(Ur))

        # UrとUiそれぞれのエッジの全ての組合せを調べる
        for e1, e2 in itertools.product(Ur, pathToRoute(Ui)):
            w1 = -cost[e1[0]][e1[1]] -cost[e2[0]][e2[1]] + \
            cost[e1[0]][e2[0]] + cost[e1[1]][e2[1]]

            w2 = -cost[e1[0]][e1[1]] -cost[e2[0]][e2[1]] + \
            cost[e1[0]][e2[1]] + cost[e1[1]][e2[0]]

            w = min(w1, w2)

            if w < best:
                best = w
                idx = 1 if w1 == min(w1, w2) else 2
                rme1 = e1
                rme2 = e2
                if(idx == 1):
                    adde1 = [e1[0], e2[0]]
                    adde2 = [e1[1], e2[1]]
                else:
                    adde1 = [e1[0], e2[1]]
                    adde2 = [e1[1], e2[0]]

        # 全ての組合せから-w(e)-w(e')+w(e")+w(e''')を最小にする
        # e∈Urとe∈Uj(j≠r)を探す
        intermediate.remove(rme1)
        intermediate.remove(rme2)
        intermediate.append(adde1)
        intermediate.append(adde2)

        subtourIndex = isHeiro(routeToPath(intermediate))

    return intermediate



def isRoute(edgeList, v_e_1, v_e):
    # エッジリストの長さが偶数かどうか
    if len(edgeList) % 2 == 0:
        if v_e_1 == v_e:
            return 1
    else:
        return 0

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
        edge_labels[(e[0], e[1])] = cost[e[0]][e[1]]

    for i in range(num_shelter):
        # labels[i] = df.ix[i].d
        labels[i] = i

    G.add_nodes_from(N)
    G.add_edges_from(E)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color="r")
    nx.draw_networkx_edges(G, pos, width=1)
    # nx.draw_networkx(G, pos, with_labels=False, node_color='r', node_size=80) # デフォルト200
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6) # デフォルト12
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6) # デフォルト8

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 70)
    plt.ylim(0, 70)
    # plt.axis('off')
    # plt.grid()

    # 元の経路
    if isFirst == 1:
        print("最初の経路:{}".format(penaltyFunction(edgeList, 0)))
        print("ペナルティ関数値:{}".format(penaltyFunction(edgeList, 1)))
        print("ルート数:{}".format(len(routeToPath(edgeList))))
        plt.title(title)
        plt.pause(0.01)
        plt.figure()

    # 連続プロット中
    if isLast == 0:
        plt.pause(0.01)
        plt.clf()
    else:
        plt.title(title)
        print("終わり")
        print("最終経路:{}".format(penaltyFunction(edgeList, 0)))
        print("ペナルティ関数値:{}".format(penaltyFunction(edgeList, 1)))
        print("ルート数:{}".format(len(routeToPath(edgeList))))
        plt.savefig("./output/" + filename +".png")  # save as png
        plt.show()
        return(0)





if __name__ == '__main__':
    filename = "vrpnc1"

    df = createDataFrame("./csv/Christ/", filename)
    num_shelter = 21

    cost = createCostMatrix(num_shelter)

    P_A, P_B = createList()
    print("P_A:" + str(P_A))
    print("P_B:" + str(P_B))

    E_A, total_cost_A = createEdgeSet(P_A)
    E_B, total_cost_B = createEdgeSet(P_B)

    # print("x_A")
    # print(x_A)
    # print("x_B")
    # print(x_B)
    print("Aの総移動コスト:{}".format(total_cost_A))
    print("Bの総移動コスト:{}".format(total_cost_B))

    # G_AB, edgelist = EAX(x_A, x_B)
    intermediate = EAX(E_A, E_B)
    # print(edgelist)

    graphPlot(intermediate, isFirst=0, isLast=1, title="child")
