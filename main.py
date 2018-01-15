#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)
#
# 配送経路問題をMemeticアルゴリズムを用いて解くプログラム
# Memeticプログラムは遺伝的アルゴリズムと局所探索を組合せた手法である
#
# GeneticAlgorithm.pyは遺伝子情報とその遺伝子の評価値を格納するclass
# 個体を取得する場合は.getGenom()
# 個体エッジ情報を取得する場合は.getEdge()
# 個体評価を取得する場合は.getEvaluation()

import GeneticAlgorithm as ga
import random
from decimal import Decimal
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys, time
import copy
import itertools

# 遺伝子集団の長さ
MAX_GENOM_LIST = 3
# 各両親から生成される子個体の数
MAX_CHILDREN = 5
# # 遺伝子選択数
# SELECT_GENOM = 20
# # 個体突然変異確率
# INDIVIDUAL_MUTATION = 0.1
# # 遺伝子突然変異確率
# GENOM_MUTATION = 0.1
# 繰り返す世代数
MAX_GENERATION = 10
# 使用できる車両数
VEHICLE = 1
# 車両の最大積載量
CAPACITY = 60
# セービング値の効果をコントロールする係数
LAMBDA = 1
# N_near()関数で，どこまで近くのノードに局所探索するか
NEAR = 5
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
    return ga.genom(genom_list, 0, 0)

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
    # print("demand:{}".format(demand))
    # print("distance:{}".format(distance))
    # print(drt)
    # graphPlot(pathToRoute(createEdgeSet(route)), isFirst=0, isLast=1)
    return route, drt

"""
セービング法で構築されたルートに，
デポを出発してデポに帰るようにエッジを辿る
3次元配列を生成する．
@INPUT:
    route:セービング法で構築されたルート(0は表示されていない)
@OUTPUT:
    path:ルートを辿るエッジの3次元配列(0も表示)
"""
def savingRoute(route):
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
2次元のエッジリストから，各閉路毎にエッジを持つ3次元リストに変換する
@INPUT:
    route: エッジの２次元リスト
@OUTPUT:
    Path: ルート情報を含むエッジの3次元リスト
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
        # エッジ端のどちらかに0を含むか
        if e[0] == 0 or e[1] == 0:
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
            e = random.choice(list(filter(lambda x: v_e in x, route)))
            R.append(e)
            route.remove(e)

            # eの端点のv_eでない方を新たにv_eとする
            v_e = e[0] if v_e == e[1] else e[1]
            # print("次の端点:{}".format(v_e))

            # 次の端点が0だった場合
            if(v_e == 0):
                Path.append(R)
                R = []
                find_flag = True
            # 閉路探索中に最初のノードを発見した場合
            if(heiro == True and v_e == v_e_1):
                Path.append(R)
                v_e_1 = 0
                R=[]
                find_flag = True
    # print(Path)
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

"""
テストも兼ねて作ったOX関数
ランダムに選んだ2点間で順列交叉をする
@INPUT:
    P_A: 親AのgenomClass
    P_B: 親Bの
    genomClass
@OUTPUT:
    c: 交叉によって生成した子の順列
"""
def orderCrossover(P_A, P_B):
    P_A = P_A.getGenom()
    P_B = P_B.getGenom()

    index1 = random.randint(1, len(P_A) - 1)
    index2 = random.randint(1, len(P_A) - 1)
    # 同じ数字を許容しない
    if index1 == index2:
        while(index1 == index2):
            index2 = random.randint(1, len(P_A))

    # index1がindex2より小さい値になるように交換
    if index1 > index2:
        index1, index2 = index2, index1

    c_A = P_A[index1:index2]
    c_B = [x for x in P_B if not x in c_A]
    c = c_B[:index1] + c_A[:] + c_B[index1:]

    return c


"""
与えたれたエッジのリストが閉路を満たすかを判定する
@INPUT:
    edgeList : エッジのリスト
    v_e : 最後に格納したエッジの端点
@OUTPUT:
    1 : 満たす
    0 : 満たさない
"""
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
EAXのステップ1と2を処理する関数
ステップ1: G_AB集合を生成する
ステップ2: AB-cycleによる閉路を生成する
@INPUT:
    P_A : 親AのgenomClass
    P_B : 親BのgenomClass
    ※それぞれ.getGenom() .getEdge()で遺伝子情報とエッジ情報を取得
@OUTPUT:
    C: AB-cycleのリスト
"""
def preEAX(P_A, P_B):
    E_A = P_A.getEdge() # 親Aのエッジリストを取得
    E_B = P_B.getEdge() # 親Bのエッジリストを取得

    """
    ステップ1: 親Aと親Bから集合G_ABを生成する
    """
    G_AB = [x for x in E_A + E_B if not (x in E_A and x in E_B)]

    # G_ABから[0, 2]と[2, 0]のような同じエッジを示す要素を排除
    for e1, e2 in itertools.combinations(G_AB, 2):
        if e1[0] == e2[1] and e1[1] == e2[0]:
            G_AB.remove(e1)
            G_AB.remove(e2)
    # print("G_AB:{}".format(G_AB))

    """
    ステップ2: G_AB上でAB-cycleを構築する
    """
    i = 0
    s = 0
    R_A = [x for x in G_AB if (x and x in E_A)]
    R_B = [x for x in G_AB if (x and x in E_B)]

    # print("R_A:{}".format(R_A))
    # print("R_B:{}".format(R_B))
    # print("R_Aの長さ:{}".format(len(R_A)))
    # print("R_Bの長さ:{}".format(len(R_B)))

    P = [0]
    C = []
    while(len(G_AB)):
        # 次の閉路を探索するためにフラグを初期化
        ABflag = False
        # G_AB上のE_Aによるエッジが繋がるノードをランダムに選択する
        v_e = random.choice(np.unique(R_A))
        v_e_1 = v_e #最初の端点を保持する
        # print("端点:{}".format(v_e))
        while (not(ABflag)):
            if(P[s] in R_B or s == 0):
                # 上で選択したノードv_eにつながるR_Aのエッジをeにセットする
                R_A = [x for x in R_A if (x and x in G_AB)]
                e = random.choice(list(filter(lambda x: v_e in x, R_A)))
                # print("e:{}".format(e))
                # G_ABから上で選択したエッジを取り除く
                G_AB = [x for x in G_AB if x != e]
                # print("eを除いたG_AB:{}".format(G_AB))
            else:
                R_B = [x for x in R_B if (x and x in G_AB)]
                e = random.choice(list(filter(lambda x: v_e in x, R_B)))
                # print("e:{}".format(e))
                G_AB = [x for x in G_AB if x != e]
                # print("eを除いたG_AB:{}".format(G_AB))
            # eの端点のv_eでない方を新たにv_eとする
            v_e = e[0] if v_e == e[1] else e[1]
            # print("次の端点:{}".format(v_e))
            s += 1
            # 配列Pにエッジを加える
            P.append(e)

            # PがAB-cycleを含んでいるか判定
            if(isRoute(P[1:], v_e_1, v_e)):
                C.append(P[1:])
                P = [0]
                s = 0
                R_A = [x for x in R_A if (x and x in G_AB)]
                ABflag = True
                # print("C:{}".format(C))
    print("AB-cycle:{}".format(C))
    return C


def edgeAssemblyCrossover(P_A, P_B, ABc):
    E_A = P_A.getEdge() # 親Aのエッジリストを取得
    E_B = P_B.getEdge() # 親Bのエッジリストを取得

    """
    ステップ3:E-setを構成する
    """
    E_set = random.choice(ABc) # Single戦略
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
    subtourIndex：部分巡回路のルートインデックス
@OUTPUT:
    child：子個体
"""
def EAXstep5(intermediate, subtourIndex):
    while(subtourIndex != 0):
        best = 0
        for e in intermediate:
            best += cost[e[0]][e[1]]

        Ui = routeToPath(intermediate) # エッジ集合(3次元リスト)
        subnum = random.choice(subtourIndex) # サブツアーインデックスをランダムに選ぶ
        Ur = Ui[subnum]  # 選択したサブツアー
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
    filename = "R101"
    # 避難所情報のデータフレームを生成する
    # 引数[0]:ファイルパス，[1]:ファイル名
    df = createDataFrame("./csv/", filename)
    num_shelter = len(df.index)
    num_shelter = 21
    result_df = pd.DataFrame(index=[], columns=['世代', '総移動コスト'])

    # 各避難所間の移動コスト行列を生成する
    # 2次元配列costで保持
    cost = createCostMatrix(num_shelter)

    # 第一世代の個体集団を生成
    current_generation_individual_group = []
    print("現行の個体集団")
    for i in range(MAX_GENOM_LIST):
        # current_generation_individual_group.append(createGenom(num_shelter, VEHICLE))
        sa_route, distance = savingMethod(num_shelter, cost)
        route = pathToRoute(savingRoute(sa_route))
        evaluation = penaltyFunction(route, option=1)

        current_generation_individual_group.append(ga.genom(route, evaluation, distance))
        print("個体{}:{}".format(i, current_generation_individual_group[i].getGenom()))
        print("評価{}".format(current_generation_individual_group[i].getEvaluation()))
        print("距離{}".format(current_generation_individual_group[i].getDistance()))

    graphPlot(current_generation_individual_group[0].getGenom(), isFirst=1, isLast=0, title="第1世代")

    monotonous = current_generation_individual_group[0].getDistance()
    mono_count = 0
    sys.exit()
    """
    ここまで第一世代
    この先繰り返し
    """
    for count_ in range(1, MAX_GENERATION + 1):
        #集団中の個体をランダムな順列に並べる
        order = setRondomOrder()

        if count_ == 1:
            # 現行の集団中の個体全てのエッジ情報をgenomClassに保存
            for i in range(MAX_GENOM_LIST):
                EdgeSet, total_cost = createEdgeSet(current_generation_individual_group[i].getGenom())
                current_generation_individual_group[i].setEdge(EdgeSet)
                current_generation_individual_group[i].setEvaluation(penaltyFunction(EdgeSet, option=1))

        for i in range(MAX_GENOM_LIST):
            # 集団中の全ての個体が丁度一度ずつ親P_Aとして選択される
            if i < MAX_GENOM_LIST -1:
                P_A = current_generation_individual_group[order[i]]
                P_B = current_generation_individual_group[order[i+1]]
            else:
                P_A = current_generation_individual_group[order[i]]
                P_B = current_generation_individual_group[order[0]]


            # 子解を代入するリスト
            c = []
            c_cost = []
            Eval = []
            """
            交叉：EAX
            """
            # EAXのステップ1~2を処理
            ABc = preEAX(P_A, P_B)

            # EAXのステップ3~5
            for j in range(MAX_CHILDREN):
                c.append(edgeAssemblyCrossover(P_A, P_B, ABc))
                Eval.append(penaltyFunction(c[j], option=1))
                c_cost.append(penaltyFunction(c[j], option=0))


            """
            orderCrossoverにてプログラム全体の処理確認
            """
            # 各両親に対してMAX_CHILDRENの数だけ子個体を生成する
            # for j in range(MAX_CHILDREN):
            #     c.append(orderCrossover(P_A, P_B))
            #     a, b = createEdgeSet(c[j]) # bがコスト
            #     Eval.append(penaltyFunction(a, option=1))
            #     c_cost.append(b)

            # 現在の親ペアの子の中で一番コストの低い個体が親Aよりも
            # 環境に適合している場合，親Aのインデックスを指定して入れ替える
            if min(Eval) < P_A.getEvaluation():
                min_idx = Eval.index(min(Eval))
                current_generation_individual_group[order[i]].setGenom(c[min_idx])
                current_generation_individual_group[order[i]].setEvaluation(Eval[min_idx])

        # 今世代の個体適用度を配列化する
        fits = [i.getEvaluation() for i in current_generation_individual_group]
        min_idx = fits.index(min(fits)) # 最小値を求める
        min_ = penaltyFunction(current_generation_individual_group[order[min_idx]].getEdge(), option=0)

        if monotonous == min_:
            mono_count += 1
        else:
            mono_count = 0
            monotonous = min_


        # データフレームの行を世代によって更新
        series = pd.Series([count_, min_], index=result_df.columns)
        result_df = result_df.append(series, ignore_index = True)

        print("====第{}世代====".format(count_))
        print("最も優れた個体の総移動コスト:{}".format(min_))

        graphPlot(current_generation_individual_group[min_idx].getEdge(), \
                  isFirst=0, isLast=0, title="第" + str(count_) + "世代")

        # 10世代以上適応度が変わらなかった場合
        if mono_count > 10:
            break

    min_idx = fits.index(min(fits))
    print(result_df)
    print("最も優れた個体:{}".format(current_generation_individual_group[min_idx].getGenom()))
    print("最も優れた個体の総移動コスト:{}".format(min_))

    graphPlot(current_generation_individual_group[min_idx].getEdge(), \
              isFirst=0, isLast=1, title="最終世代")
    result_df.to_csv("./output/" + filename + "_result.csv", index=False)
