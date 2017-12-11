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
import time

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
LAMBDA = 0.5


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


def Neighborhoods(v, path, neighbor):
    EdgeSet = []
    path_cost = 0

    # 解を構成する全てのエッジ集合を生成
    for edge in path:
        for j in edge:
            EdgeSet.append(j)
    print(EdgeSet)

    # 渡されたノードvに繋がるエッジ2つ
    link_v = [i for i in EdgeSet if (v in i)]
    v_minus = link_v[0][0] if v == link_v[0][1] else link_v[0][1]
    v_plus = link_v[1][1] if v == link_v[1][0] else link_v[1][0]
    # print(link_v[0]) # ノードvに向かうエッジ
    # print(link_v[1]) # ノードvを出るエッジ

    # ノードvからnearだけ近いノードをそれぞれwとして選ぶ
    for w in N_near(v, 5):

        link_w = [i for i in EdgeSet if (w in i)]

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

        # print(neighbor)

        """
        (1,0)Interchange
        """
        if neighbor == "10inter":
            # 選んだwに対して
            # 元々繋がっているエッジ
            l1 = cost[w_minus][w] # w-→w+
            l2 = cost[v_minus][v] # v-→v
            l3 = cost[v][v_plus] # v→v+
            # つなぎ直すエッジ
            l4 = cost[w_minus][v]
            l5 = cost[v][w]
            l6 = cost[v_minus][v_plus]

            if l1 + l2 + l3 > l4 + l5 + l6:
                print("(1,0)Interchangeやるよー")
                EdgeSet.remove(link_w[0]) #-を含む方
                EdgeSet.remove(link_v[0])
                EdgeSet.remove(link_v[1]) #+を含む方

                EdgeSet.append([w_minus, v])
                EdgeSet.append([v, w])
                EdgeSet.append([v_minus, v_plus])
                return EdgeSet

        """
        2-opt近傍
        """
        if neighbor == "2opt":
            """
            2-opt①
            """
            print("2opt①やるよー")
            l1 = cost[v_minus][v] # 元のエッジ
            l2 = cost[w_minus][w] # 元のエッジ
            l3 = cost[v][w]
            l4 = cost[v_minus][w_minus]

            if l1 + l2 > l3 + l4:
                print("2-opt① " + str(v) + ":" + str(w) + "適用")
                EdgeSet.remove(link_w[0])
                EdgeSet.remove(link_v[0])
                EdgeSet.append([v, w])
                EdgeSet.append([v_minus, w_minus])
                return EdgeSet

            """
            2-opt④
            """
            print("2opt④やるよー")
            l1 = cost[v][v_plus] # 元のエッジ
            l2 = cost[w][w_plus] # 元のエッジ
            l3 = cost[v][w]
            l4 = cost[v_plus][w_plus]

            if l1 + l2 > l3 + l4:
                print("2-opt④ " + str(v) + ":" + str(w) + "適用")
                EdgeSet.remove(link_v[1])
                EdgeSet.remove(link_w[1])
                EdgeSet.append([v, w])
                EdgeSet.append([v_plus, w_plus])
                return EdgeSet

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
                print("2opt②やるよー")
                l1 = cost[v_minus][v] # 元のエッジ
                l2 = cost[w][w_plus] # 元のエッジ
                l3 = cost[v][w]
                l4 = cost[v_minus][w_plus]

                if l1 + l2 > l3 + l4:
                    print("2-opt② " + str(v) + ":" + str(w) + "適用")
                    EdgeSet.remove(link_v[0])
                    EdgeSet.remove(link_w[1])
                    EdgeSet.append([v, w])
                    EdgeSet.append([v_minus, w_plus])
                    return EdgeSet

                """
                2-opt③
                """
                print("2opt③やるよー")
                l1 = cost[v][v_plus] # 元のエッジ
                l2 = cost[w_minus][w] # 元のエッジ
                l3 = cost[v][w]
                l4 = cost[v_plus][w_minus]

                if l1 + l2 > l3 + l4:
                    print("2-opt③ " + str(v) + ":" + str(w) + "適用")
                    EdgeSet.remove(link_v[1])
                    EdgeSet.remove(link_w[0])
                    EdgeSet.append([v, w])
                    EdgeSet.append([v_plus, w_minus])
                    return EdgeSet

    return EdgeSet

"""
2次元のエッジリストから，各閉路毎にエッジを持つ3次元リストに変換する
"""
def routeToPath(route):
        route = sorted(route)
        # print(route)
        Path = [[[]]]
        find_flag = False

        for e in route:
            find_flag = False
            # 最初のエッジをリストに格納
            if Path == [[[]]]:
                Path[0][0] = e
                continue

            for i, k in enumerate(Path):
                if(find_flag):
                    break
                if ((e[0] != 0 and e[0] in np.unique(k)) or \
                    (e[1] != 0 and e[1] in np.unique(k))):
                    Path[i].append(e)
                    break

                if i == len(Path)-1:
                    Path.append([e])
                    find_flag = True
                    break
        print(Path)
        return Path






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


    E = test
    print(E)
    G.add_nodes_from(N)
    G.add_edges_from(E)
    nx.draw_networkx(G, pos, with_labels=False, node_color='r', node_size=80) # デフォルト200
    nx.draw_networkx_labels(G, pos, labels, font_size=6) # デフォルト12
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6) # デフォルト8

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 70)
    plt.ylim(0, 70)
    # plt.axis('off')
    plt.title('Delivery route')
    plt.savefig("./output/" + filename +".png")  # save as png
    # plt.grid()
    plt.show()

    return(0)


if __name__ == "__main__":
    filename = "R101"

    df = createDataFrame("./csv/", filename)
    num_shelter = len(df.index)
    num_shelter = 11

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

    path = createEdgeSet(route)
    # localSearch(path)
    test = Neighborhoods(7, path, "2opt")

    print(test)
    routeToPath(test)



    # for i in range(1, num_shelter):
    #     local_route = Neighborhoods(i, path, "2opt")
    #     path = routeToPath(local_route)


    # EdgeSet = []
    # for edge in path:
    #     for j in edge:
    #         EdgeSet.append(j)
    # test = EdgeSet


    # X, Y, N, pos, G = createGraphList()  #グラフ描画準備
    # graphPlot(G, N, route)
