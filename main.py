#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)
#
# 配送経路問題をMemeticアルゴリズムを用いて解くプログラム
# Memeticプログラムは遺伝的アルゴリズムと局所探索を組合せた手法である
#
# GeneticAlgorithm.pyは遺伝子情報とその遺伝子の評価値を格納するclass
# 値を取得する場合は.getGenom()

import GeneticAlgorithm as ga
import random
from decimal import Decimal
import numpy as np
import pandas as pd

# 遺伝子情報の長さ
GENOM_LENGTH = 50
# 遺伝子集団の長さ
MAX_GENOM_LIST = 2
# 各両親から生成される子個体の数
MAX_CHILDREN = 1
# 遺伝子選択数
SELECT_GENOM = 20
# 個体突然変異確率
INDIVIDUAL_MUTATION = 0.1
# 遺伝子突然変異確率
GENOM_MUTATION = 0.1
# 繰り返す世代数
MAX_GENERATION = 2
# 使用できる車両数
VEHICLE = 3



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
    x : 移動するエッジの行列
"""
def createEdgeMatrix(ga):
    # 配送順序の配列を変数genomにコピー
    genom = ga.getGenom()
    total_cost = 0
    route_flag = False
    # どの避難所間を通ったかを示す2次元配列を0で初期化
    x = np.zeros((num_shelter, num_shelter), int) #小数点以下を加える→float型
    for i in range(len(genom)):
        # ルート区切り番号だった場合
        if genom[i] > num_shelter - 1: # >10
            if route_flag == True:
                x[genom[i-1]][0] = 1
                # print("{}→{}".format(genom[i-1], 0))
            route_flag = False
            # print("_{}_区切り".format(genom[i]))
        else : # ルート区切り番号ではない場合(避難所番号)

            # 現在参照している避難所番号の前が区切り番号だった，
            # もしくは遺伝子の最初を参照している場合
            if route_flag == False:
                x[0][genom[i]] = 1
                route_flag = True
                # print("{}→{}".format(0, genom[i]))
            else : # フラグがTrue，つまり経路続行
                x[genom[i-1]][genom[i]] = 1
                # print("{}→{}".format(genom[i-1], genom[i]))
    # 遺伝子の最後の番号が区切り番号でない場合，
    if route_flag == True:
        x[genom[i]][0] = 1
        # print("{}→{}".format(genom[i], 0))

    #総移動コストの計算
    for i in range(num_shelter):
        for j in range(num_shelter):
            total_cost += cost[i][j] * x[i][j]

    # print("総移動コスト:{}".format(total_cost))
    # 総移動コストと，移動エッジ行列を返す
    return x

"""
【家族内淘汰】
親P_Aとその子の中から一番良い個体を選択する
@INPUT:
    ga : 選択を行うgenomClassの配列
@OUTPUT:
    選択処理をした一定のエリートgenomClass
"""
def select(ga, elite):
    pass


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
EAXのステップ1と2を処理する関数
ステップ1: G_AB集合を生成する
ステップ2: AB-cycleによる閉路を生成する
@INPUT:
    P_A : 親AのgenomClass
    P_B : 親BのgenomClass
    ※それぞれ.getGenom() .getEdge()で遺伝子情報とエッジ情報を取得
@OUTPUT:

"""
def preEAX(P_A, P_B):
    E_A = [] #親Aのエッジを格納するリスト
    E_B = [] #親Bのエッジを格納するリスト
    x_A = P_A.getEdge() # 親Aの遺伝子リストを取得
    x_B = P_B.getEdge() # 親Bの遺伝子リストを取得
    x_AB = np.zeros((num_shelter, num_shelter), int) #小数点以下を加える→float型

    # エッジがある行列番号をE_A[]とE_B[]に追加
    for i in range(num_shelter):
        for j in range(i, num_shelter):
            if(x_A[i][j] == 1 or x_A[j][i] == 1):
                E_A.append([i, j])
            if(x_B[i][j] == 1 or x_B[j][i] == 1):
                E_B.append([i, j])

    """
    ステップ1: 親Aと親Bから集合G_ABを生成する
    """
    G_AB = sorted([x for x in E_A + E_B if not (x in E_A and x in E_B)])
    # print(G_AB)
    """
    ステップ2: G_AB上でAB-cycleを構築する
    """
    i = 0
    s = 0
    R_A = sorted([x for x in G_AB if (x and x in E_A)])
    R_B = sorted([x for x in G_AB if (x and x in E_B)])

    print("G_AB:{}".format(G_AB))
    print("R_A:{}".format(R_A))
    print("R_B:{}".format(R_B))

    # G_AB = [[0, 6], [0, 8], [0, 9], [0, 10], [2, 4], [2, 6], [3, 4], [3, 9], [7, 8], [7, 10]]
    # R_A = [[0, 6], [0, 10], [2, 4], [3, 9], [7, 8]]
    # R_B = [[0, 8], [0, 9], [2, 6], [3, 4], [7, 10]]
    P = [0]
    C = []

    while(len(G_AB)):
        # 次の閉路を探索するためにフラグを初期化
        ABflag = False
        # G_AB上のE_Aによるエッジが繋がるノードをランダムに選択する
        v_e = random.choice(np.unique(R_A))
        v_e_1 = v_e #最初の端点を保持する
        print("端点:{}".format(v_e))
        while (not(ABflag)):
            if(P[s] in R_B or s == 0):
                # 上で選択したノードv_eにつながるR_Aのエッジをeにセットする
                R_A = sorted([x for x in R_A if (x and x in G_AB)])
                e = random.choice(list(filter(lambda x: v_e in x, R_A)))
                print("e:{}".format(e))
                # G_ABから上で選択したエッジを取り除く
                G_AB = [x for x in G_AB if x != e]
                print("eを除いたG_AB:{}".format(G_AB))
            else:
                R_B = sorted([x for x in R_B if (x and x in G_AB)])
                e = random.choice(list(filter(lambda x: v_e in x, R_B)))
                print("e:{}".format(e))
                G_AB = [x for x in G_AB if x != e]
                print("eを除いたG_AB:{}".format(G_AB))
            # eの端点のv_eでない方を新たにv_eとする
            v_e = e[0] if v_e == e[1] else e[1]
            print("次の端点:{}".format(v_e))
            s += 1
            # 配列Pにエッジを加える
            P.append(e)

            # PがAB-cycleを含んでいるか判定
            if(isRoute(P[1:], v_e_1, v_e)):
                C.append(P[1:])
                P = [0]
                s = 0
                R_A = sorted([x for x in R_A if (x and x in G_AB)])
                ABflag = True
                print("C:{}".format(C))
    for i, x in enumerate(C):
        print("C{}:{}".format(i, x))


def edgeAssemblyCrossover():
    pass



if __name__ == '__main__':

    # 避難所情報のデータフレームを生成する
    # 引数[0]:ファイルパス，[1]:ファイル名
    df = createDataFrame("./data/", "data_r101")
    #num_shelter = len(df.index)
    num_shelter = 11

    # 各避難所間の移動コスト行列を生成する
    # 2次元配列costで保持
    cost = createCostMatrix(num_shelter)

    # 第一世代の個体集団を生成
    current_generation_individual_group = []
    print("現行の個体集団")
    for i in range(MAX_GENOM_LIST):
        current_generation_individual_group.append(createGenom(num_shelter, VEHICLE))
        print(current_generation_individual_group[i].getGenom())
    """
    ここまで第一世代
    この先繰り返し
    """
    for count_ in range(1, MAX_GENERATION + 1):
        #集団中の個体をランダムな順列に並べる
        order = setRondomOrder()

        # 現行の集団中の個体全てのエッジ情報をgenomClassに保存
        for i in range(MAX_GENOM_LIST):
            x = createEdgeMatrix(current_generation_individual_group[i])
            current_generation_individual_group[i].setEdge(x)

        for i in range(MAX_GENOM_LIST):
            # 集団中の全ての個体が丁度一度ずつ親P_Aとして選択される
            if i < MAX_GENOM_LIST -1:
                P_A = current_generation_individual_group[order[i]]
                P_B = current_generation_individual_group[order[i+1]]
            else:
                P_A = current_generation_individual_group[order[i]]
                P_B = current_generation_individual_group[order[0]]

            # EAXのステップ1~2を処理する
            preEAX(P_A, P_B)
            # 各両親に対してMAX_CHILDRENの数だけ子個体を生成する
            for j in range(MAX_CHILDREN):
                edgeAssemblyCrossover() #GAクラスに子個体情報も持たせる

        """
        #現行世代個体集団の遺伝子を評価し，genomClassに代入
        for i in range(MAX_GENOM_LIST):
            evaluation_result, x = evaluation(current_generation_individual_group[i])
            current_generation_individual_group[i].setEvaluation(evaluation_result)
            current_generation_individual_group[i].setEdge(x) # 移動エッジ行列をgenomClassに保存
        # print("エッジ確認")
        # print(current_generation_individual_group[4].getEdge())
        """

        # #遺伝子集団それぞれの評価値確認
        # print("====第{}世代====".format(count_))
        # for i in range(MAX_GENOM_LIST):
        #     print("遺伝子<{}>:{}".format(i + 1, current_generation_individual_group[i].getEvaluation()))

        #エリート個体を選択する
        #elite_genes = select(current_generation_individual_group, SELECT_GENOM)
