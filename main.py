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
# from Local import Neighborhoods as local
import random
from decimal import Decimal
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys, time
import copy
import itertools
import math
import csv

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
MAX_GENERATION = 100
# 使用できる車両数
m = 8
# 車両の最大積載量
CAPACITY = 160
# セービング値の効果をコントロールする係数
LAMBDA = 1
# N_near()関数で，どこまで近くのノードに局所探索するか
NEAR = 10
# penaltyFunction()で，容量制約違反に課すペナルティの係数
ALPHA = 0.003
# penaltyFunction()で，経路長違反に課すペナルティの係数
BETA = 1.0
# penaltyFunction()で，経路長違反とする距離
D = float("inf")

"""
近傍操作関数
@INPUT:
    v: 近傍操作対象ノード
    path: 解の3次元リスト
    f_option: ペナルティ関数をどのように評価するか
        2: 容量超過
        1: ペナルティ関数による評価
        0: 総移動コストのみの評価
    reduce_route: 経路数が減る遷移を許容するか
        1: 許容する
        0: 許容しない
"""
def Local(v, path, f_option, reduce_route):
    EdgeSet = []
    # DefaultEdgeSet = []
    path_cost = 0
    bestSet = []
    flag = False

    # 解を構成する全てのエッジ集合を生成
    for edge in path:
        for j in edge:
            EdgeSet.append(j)
    # DefaultEdgeSet = copy.deepcopy(EdgeSet)
    bestSet = copy.deepcopy(EdgeSet)
    # print("デフォ:{}".format(DefaultEdgeSet))

    # 元の解のペナルティ関数評価値を保持
    P_eval = penaltyFunction(bestSet, f_option)
    # print("元の評価関数値:{}".format(P_eval))

    # 渡されたノードvに繋がるエッジ2つ
    link_v = [i for i in EdgeSet if (v in i)]
    v_minus = link_v[0][0] if v == link_v[0][1] else link_v[0][1]
    v_plus = link_v[1][1] if v == link_v[1][0] else link_v[1][0]
    # print("link_v:{}".format(link_v))
    # print("link_v[0]:{}".format(link_v[0])) # ノードvに向かうエッジ
    # print("link_v[1]:{}".format(link_v[1])) # ノードvを出るエッジ

    # ノードvからnearだけ近いノードをそれぞれwとして選ぶ
    for w in N_near(v, NEAR):
        w = int(w)
        link_w = [i for i in EdgeSet if (w in i)]

        if(len(link_v) < 2 or len(link_w) < 2):
            # print("ノードに対するエッジが2つない")
            EdgeSet = copy.deepcopy(bestSet)
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
        # if neighbor == "10inter":
        # print("(1,0)Interchange①やるよー")
        """
        (1,0)Interchange①
        """
        # 選んだwに対して
        # 元々繋がっているエッジ
        l1 = cost[w_minus][w] # w-→w
        l2 = cost[v_minus][v] # v-→v
        l3 = cost[v][v_plus] # v→v+
        # # つなぎ直すエッジ
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
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    # ルート数が減る遷移をどうするか
                    if es[0] == 0 and reduce_route == 1:
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False
        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 + l3 > l4 + l5 + l6):
                        print("【Neighborhoods】修正操作１")
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)

            # ペナルティ関数値が元より大きい
            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet

        flag = False
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
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    if es[0] == 0 and reduce_route == 1:
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False
        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 + l3 > l4 + l5 + l6):
                        print("【Neighborhoods】修正操作１")
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)

            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet
        flag = False

        """
        (0,1)Interchange
        """
        # if neighbor == "01inter":
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
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1]:
                    # ルート数が減る遷移をどうするか
                    if es[0] == 0 and reduce_route == 1:
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False

        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 + l3 > l4 + l4 + l5):
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)
            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet
        flag = False

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
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            # [n, n]のようなエッジを持たないように修正
            for es in EdgeSet:
                if es[0] == es[1] and reduce_route == 1:
                    if es[0] == 0:
                        # ルート数が減る遷移をどうするか
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False

        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(DefaultEdgeSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 + l3 > l4 + l5 + l6):
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)
            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet
        flag = False

        """
        (1,1)Interchange
        """
        # if neighbor == "11inter":
            # print("(1,1)Interchangeやるよー")
        """
        (1,1)Interchange①
        """

        """
        w--を求めるステップ
        """
        path = routeToPath(EdgeSet)
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

        node_flag = False
        try:
            # リストからノードw-でない方をノードw--として選択
            w_minus_minus = w_minus2list[0][0] if w_minus2list[0][1] == w_minus \
            else w_minus2list[0][1]
            node_flag = True
        except IndexError:
            print("w--作成で失敗")
            node_flag = False

        # 同じエッジを選ぶ可能性があるものは排除
        # if link_v[0] == w_minus2list[0] or link_v[1] == w_minus2list[0] \
        # or link_w[0] == w_minus2list[0] or link_w[1] == w_minus2list[0]:
        #     continue
        if node_flag == True:
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
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1] and reduce_route == 1:
                        if es[0] == 0:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False

            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    # print("リセット1")
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 + l3 + l4 > l5 + l6 + l7 + l8):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)
                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    return bestSet
            flag = False

        """
        (1,1)Interchange②
        """

        """
        w++を求めるステップ
        """
        # ノードw+を持つリストを作る
        w_plus2list = [x for x in path[route_num] \
                        if(w_plus in x and x != link_w[1])]

        node_flag = False
        try:
            # リストからノードw+でない方をノードw++として選択
            w_plus_plus = w_plus2list[0][1] if w_plus2list[0][0] == w_plus \
            else w_plus2list[0][0]
            node_flag = True
        except IndexError:
            print("w++の作成に失敗")
            node_flag = False

        # print("{}を含むルートは:{}".format(w, path[route_num]))
        # print("リスト:{}".format(w_plus2list[0]))
        # print("w+:{}".format(w_plus))
        # print("w++:{}".format(w_plus_plus))
        if node_flag == True:
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
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1] and reduce_route == 1:
                        if es[0] == 0:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False

            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    # print("リセット２")
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 + l3 + l4 > l5 + l6 + l7 + l8):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)
                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    return bestSet
            flag = False


        """
        2-opt近傍
        """
        # if neighbor == "2opt":
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
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            for es in EdgeSet:
                if es[0] == es[1]:
                    if es[0] == 0 and reduce_route == 1:
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False

        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 > l3 + l4):
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)
            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet
        flag = False
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
            flag = True
        except ValueError:
            EdgeSet = copy.deepcopy(bestSet)
            flag = False

        if flag == True:
            for es in EdgeSet:
                if es[0] == es[1]:
                    if es[0] == 0 and reduce_route == 1:
                        EdgeSet.remove([0, 0])
                    else:
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False

        if flag == True:
            # デポを含まない巡回路ができた場合
            if(isHeiro(routeToPath(EdgeSet)) != 0):
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

        if flag == True:
            current_function = penaltyFunction(EdgeSet, f_option)
            # ペナルティ関数により評価
            if(f_option == 2):
                # 元の解も改良解も実行可能な場合，距離の短い方を返す
                if P_eval <= 0 and current_function <= 0:
                    if(l1 + l2 > l3 + l4):
                        return EdgeSet
                    else:
                        EdgeSet = copy.deepcopy(bestSet)

                if(current_function < P_eval):
                    return EdgeSet
                EdgeSet = copy.deepcopy(bestSet)

            elif current_function > P_eval:
                EdgeSet = copy.deepcopy(bestSet)
            else:
                P_eval = current_function
                bestSet = copy.deepcopy(EdgeSet)
                return bestSet
        flag = False

        # vとwが同じルートか判定
        sameRoute = False
        path = routeToPath(EdgeSet)
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
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                for es in EdgeSet:
                    if es[0] == es[1]:
                        if es[0] == 0 and reduce_route == 1:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False

            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 > l3 + l4):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)
                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    return bestSet
            flag = False
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
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1]:
                        if es[0] == 0 and reduce_route == 1:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False
            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 > l3 + l4):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)

                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    return bestSet
    return bestSet

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
    heiro = []

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
    drt = 0.00
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
経路を分割することでルート数を増やす関数
セービング法によって生成された初期解のうち，
もっとも顧客数の多いルートを選び真ん中で分割する
@INPUT:
    path
    (m): 固定したいルート数
@OUTPUT:
    path
"""
def routeSplit(path):
    routeLength = []
    routeCost = []
    routeCapa = []
    s_option = S_OP
    # 0: ランダム
    # 1: エッジ数
    # 2: 距離
    # 3: 需要量
    # print("パス:{}".format(path))
    while(True):
        for route in path:
            r_cost = 0
            r_demand = 0
            routeLength.append(len(route)) # ルート毎の顧客数リスト
            for n in np.unique(route):
                r_demand += df.ix[n].d
            for e in route:
                r_cost += cost[int(e[0])][int(e[1])]
            routeCapa.append(r_demand) # ルート毎の需要量
            routeCost.append(round(r_cost, 2)) # ルート毎の距離リスト

        # print("routeLength:{}".format(routeLength))
        # print("routeCapa:{}".format(routeCapa))
        # print("routeCost:{}".format(routeCost))

        # エッジ数6以上のところからランダムに選ぶ
        if s_option == 0:
            IdxList = [i for i, x in enumerate(routeLength) if x >= 6]
            # print("routeLength:{}".format(routeLength))
            # print(IdxList)
        # エッジ数の多いルートを選ぶ
        elif s_option == 1:
            IdxList = [i for i, x in enumerate(routeLength) if x == max(routeLength)]
            # print("エッジ数:{}".format(routeLength))
            # print(IdxList)
        # ルート内の距離が一番長いルートを選ぶ
        elif s_option == 2:
            IdxList = [i for i, x in enumerate(routeCost) if x == max(routeCost)]
            # print("距離:{}".format(routeCost))
            # print(IdxList)
        # ルート内の需要が一番多いルートを選ぶ
        elif s_option == 3:
            IdxList = [i for i, x in enumerate(routeCapa) if x == max(routeCapa)]
            # print("需要量:{}".format(routeCapa))
            # print(IdxList)

        maxIdx = random.choice(IdxList)
        # print("maxIdx:{}".format(maxIdx))
        # print("それぞれの顧客数:{}".format(routeLength))

        # 分割するルート
        split = path[maxIdx]
        # 分割するルートを除いたパス
        path.pop(maxIdx)
        # print("split:{}".format(split))
        # print("残ったパス:{}".format(path))
        # 分割するインデックス
        Idx = math.floor(len(split)/2 - 1)
        # print("Idx:{}".format(Idx))

        #分岐点となるノードを決定する
        pointA = split[Idx][1] if split[Idx][1] in split[Idx] and split[Idx][1] in split[Idx+1] \
        else split[Idx][0]
        pointB = split[Idx+1][1] if split[Idx+1][0] == pointA else split[Idx+1][0]

        routeA = split[:Idx+1] + [[pointA, 0]]
        routeB = [[0, pointB]] + split[Idx+2:]

        # print("routeA:{}".format(routeA))
        # print("routeB:{}".format(routeB))

        path.append(routeA)
        path.append(routeB)

        # print("分割後パス:{}".format(path))

        if len(path) >= m:
            break

        routeLength = []
        routeCost = []
        routeCapa = []
    return path

"""
2次元のエッジリストから，各閉路毎にエッジを持つ3次元リストに変換する
@INPUT:
    route: エッジの２次元リスト
@OUTPUT:
    Path: ルート情報を含むエッジの3次元リスト
"""
def routeToPath(pre_route):

    for n in pre_route:
        if n[1] == 0:
            n[0], n[1] = n[1], n[0]

    route = sorted(pre_route)
    # route = sorted(route)

    Path = []
    R = []
    v_e_1 = 0

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
def Penalty(route, option):
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
    F_propose = 0
    R_demands = 0
    R_cost = 0
    path = routeToPath(route)
    capaList = []

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
            F_propose += R_demands - CAPACITY
            F_propose *= float("inf")
        else:
            F_c += 0
            F_propose += abs(R_demands - CAPACITY)
            # F_c += R_demands - CAPACITY # ルート内の需要超過
        capaList.append(F_propose)
        F_propose = 0
        R_demands = 0


    ave = sum(capaList)/len(capaList)
    # print(capaList)
    # print("平均:{}".format(ave))

    value = 0
    for capa in capaList:
        # value += np.power((capa - ave), 2)
        value += abs(capa - ave)

    # print(value)
    # ペナルティ関数
    # F_p = F + (ALPHA * F_c) + (BETA * F_d)
    # F_p = F + (ALPHA * value)
    F_p = value

    # print("F:{}, F_c:{}, F_d:{}".format(F, F_c, F_d))
    if option == 1:
        return round(F_p, 2) # ペナルティ間数値
    elif option == 2:
        # print("ペナルティ項の合計:{}".format(round(F_c + F_d, 2)))
        return round(F_c + F_d, 2) # ペナルティ項のみ


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
    for i, c in enumerate(cost[int(node)][:]):
        if i == 0:
            continue
        near_cost = np.append(near_cost, np.array([[i, c]]), axis=0)
    near_cost = near_cost[near_cost[:, 1].argsort()] # nodeに近い順にソート
    # print(near_cost)

    # print(near_cost[1:near+1, 0])
    return near_cost[1:near+1, 0]


"""
近傍操作関数
@INPUT:
    v: 近傍操作対象ノード
    path: 解の3次元リスト
    f_option: ペナルティ関数をどのように評価するか
        2: 容量超過
        1: ペナルティ関数による評価
        0: 総移動コストのみの評価
    reduce_route: 経路数が減る遷移を許容するか
        1: 許容する
        0: 許容しない
"""
def Neighborhoods(v, path, f_option, reduce_route, first):
    EdgeSet = []
    # DefaultEdgeSet = []
    path_cost = 0
    bestSet = []
    flag = False

    # 解を構成する全てのエッジ集合を生成
    for edge in path:
        for j in edge:
            EdgeSet.append(j)
    # DefaultEdgeSet = copy.deepcopy(EdgeSet)
    bestSet = copy.deepcopy(EdgeSet)
    # print("デフォ:{}".format(DefaultEdgeSet))

    # 元の解のペナルティ関数評価値を保持
    P_eval = penaltyFunction(bestSet, f_option)
    # print("元の評価関数値:{}".format(P_eval))

    # 渡されたノードvに繋がるエッジ2つ
    link_v = [i for i in EdgeSet if (v in i)]
    v_minus = link_v[0][0] if v == link_v[0][1] else link_v[0][1]
    v_plus = link_v[1][1] if v == link_v[1][0] else link_v[1][0]
    # print("link_v:{}".format(link_v))
    # print("link_v[0]:{}".format(link_v[0])) # ノードvに向かうエッジ
    # print("link_v[1]:{}".format(link_v[1])) # ノードvを出るエッジ

    # ノードvからnearだけ近いノードをそれぞれwとして選ぶ
    for w in N_near(v, NEAR):
        w = int(w)
        link_w = [i for i in EdgeSet if (w in i)]

        if(len(link_v) < 2 or len(link_w) < 2):
            # print("ノードに対するエッジが2つない")
            EdgeSet = copy.deepcopy(bestSet)
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
        追加分
        """
        # vとwが同じルートか判定
        sameRoute = False
        path = routeToPath(EdgeSet)
        for r in path:
            node = np.unique(r)
            if v in node and w in node:
                sameRoute = True

        #ノードvとwが異なるルートに存在する時
        if sameRoute:

            """
            (1,0)Interchange
            """
            # if neighbor == "10inter":
            # print("(1,0)Interchange①やるよー")
            """
            (1,0)Interchange①
            """
            # 選んだwに対して
            # 元々繋がっているエッジ
            l1 = cost[w_minus][w] # w-→w
            l2 = cost[v_minus][v] # v-→v
            l3 = cost[v][v_plus] # v→v+
            # # つなぎ直すエッジ
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
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1]:
                        # ルート数が減る遷移をどうするか
                        if es[0] == 0 and reduce_route == 1:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False
            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 + l3 > l4 + l5 + l6):
                            print("【Neighborhoods】修正操作１")
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)

                # ペナルティ関数値が元より大きい
                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    if first == 1:
                        return bestSet

            flag = False
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
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1]:
                        if es[0] == 0 and reduce_route == 1:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False
            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 + l3 > l4 + l5 + l6):
                            print("【Neighborhoods】修正操作１")
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)

                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    if first == 1:
                        return bestSet
            flag = False

            """
            (0,1)Interchange
            """
            # if neighbor == "01inter":
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
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1]:
                        # ルート数が減る遷移をどうするか
                        if es[0] == 0 and reduce_route == 1:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False

            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 + l3 > l4 + l4 + l5):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)
                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    if first == 1:
                        return bestSet
            flag = False

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
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                # [n, n]のようなエッジを持たないように修正
                for es in EdgeSet:
                    if es[0] == es[1] and reduce_route == 1:
                        if es[0] == 0:
                            # ルート数が減る遷移をどうするか
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False

            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(DefaultEdgeSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 + l3 > l4 + l5 + l6):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)
                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    if first == 1:
                        return bestSet
            flag = False

            """
            (1,1)Interchange
            """
            # if neighbor == "11inter":
                # print("(1,1)Interchangeやるよー")
            """
            (1,1)Interchange①
            """

            """
            w--を求めるステップ
            """
            path = routeToPath(EdgeSet)
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

            node_flag = False
            try:
                # リストからノードw-でない方をノードw--として選択
                w_minus_minus = w_minus2list[0][0] if w_minus2list[0][1] == w_minus \
                else w_minus2list[0][1]
                node_flag = True
            except IndexError:
                print("w--作成で失敗")
                node_flag = False

            # 同じエッジを選ぶ可能性があるものは排除
            # if link_v[0] == w_minus2list[0] or link_v[1] == w_minus2list[0] \
            # or link_w[0] == w_minus2list[0] or link_w[1] == w_minus2list[0]:
            #     continue
            if node_flag == True:
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
                    flag = True
                except ValueError:
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

                if flag == True:
                    # [n, n]のようなエッジを持たないように修正
                    for es in EdgeSet:
                        if es[0] == es[1] and reduce_route == 1:
                            if es[0] == 0:
                                EdgeSet.remove([0, 0])
                            else:
                                EdgeSet = copy.deepcopy(bestSet)
                                flag = False

                if flag == True:
                    # デポを含まない巡回路ができた場合
                    if(isHeiro(routeToPath(EdgeSet)) != 0):
                        # print("リセット1")
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False

                if flag == True:
                    current_function = penaltyFunction(EdgeSet, f_option)
                    # ペナルティ関数により評価
                    if(f_option == 2):
                        # 元の解も改良解も実行可能な場合，距離の短い方を返す
                        if P_eval <= 0 and current_function <= 0:
                            if(l1 + l2 + l3 + l4 > l5 + l6 + l7 + l8):
                                return EdgeSet
                            else:
                                EdgeSet = copy.deepcopy(bestSet)

                        if(current_function < P_eval):
                            return EdgeSet
                        EdgeSet = copy.deepcopy(bestSet)
                    elif current_function > P_eval:
                        EdgeSet = copy.deepcopy(bestSet)
                    else:
                        P_eval = current_function
                        bestSet = copy.deepcopy(EdgeSet)
                        if first == 1:
                            return bestSet
                flag = False

            """
            (1,1)Interchange②
            """

            """
            w++を求めるステップ
            """
            # ノードw+を持つリストを作る
            w_plus2list = [x for x in path[route_num] \
                            if(w_plus in x and x != link_w[1])]

            node_flag = False
            try:
                # リストからノードw+でない方をノードw++として選択
                w_plus_plus = w_plus2list[0][1] if w_plus2list[0][0] == w_plus \
                else w_plus2list[0][0]
                node_flag = True
            except IndexError:
                print("w++の作成に失敗")
                node_flag = False

            # print("{}を含むルートは:{}".format(w, path[route_num]))
            # print("リスト:{}".format(w_plus2list[0]))
            # print("w+:{}".format(w_plus))
            # print("w++:{}".format(w_plus_plus))
            if node_flag == True:
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
                    flag = True
                except ValueError:
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

                if flag == True:
                    # [n, n]のようなエッジを持たないように修正
                    for es in EdgeSet:
                        if es[0] == es[1] and reduce_route == 1:
                            if es[0] == 0:
                                EdgeSet.remove([0, 0])
                            else:
                                EdgeSet = copy.deepcopy(bestSet)
                                flag = False

                if flag == True:
                    # デポを含まない巡回路ができた場合
                    if(isHeiro(routeToPath(EdgeSet)) != 0):
                        # print("リセット２")
                        EdgeSet = copy.deepcopy(bestSet)
                        flag = False

                if flag == True:
                    current_function = penaltyFunction(EdgeSet, f_option)
                    # ペナルティ関数により評価
                    if(f_option == 2):
                        # 元の解も改良解も実行可能な場合，距離の短い方を返す
                        if P_eval <= 0 and current_function <= 0:
                            if(l1 + l2 + l3 + l4 > l5 + l6 + l7 + l8):
                                return EdgeSet
                            else:
                                EdgeSet = copy.deepcopy(bestSet)

                        if(current_function < P_eval):
                            return EdgeSet
                        EdgeSet = copy.deepcopy(bestSet)
                    elif current_function > P_eval:
                        EdgeSet = copy.deepcopy(bestSet)
                    else:
                        P_eval = current_function
                        bestSet = copy.deepcopy(EdgeSet)
                        if first == 1:
                            return bestSet
                flag = False


            """
            2-opt近傍
            """
            # if neighbor == "2opt":
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
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                for es in EdgeSet:
                    if es[0] == es[1]:
                        if es[0] == 0 and reduce_route == 1:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False

            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 > l3 + l4):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)
                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    if first == 1:
                        return bestSet
            flag = False
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
                flag = True
            except ValueError:
                EdgeSet = copy.deepcopy(bestSet)
                flag = False

            if flag == True:
                for es in EdgeSet:
                    if es[0] == es[1]:
                        if es[0] == 0 and reduce_route == 1:
                            EdgeSet.remove([0, 0])
                        else:
                            EdgeSet = copy.deepcopy(bestSet)
                            flag = False

            if flag == True:
                # デポを含まない巡回路ができた場合
                if(isHeiro(routeToPath(EdgeSet)) != 0):
                    EdgeSet = copy.deepcopy(bestSet)
                    flag = False

            if flag == True:
                current_function = penaltyFunction(EdgeSet, f_option)
                # ペナルティ関数により評価
                if(f_option == 2):
                    # 元の解も改良解も実行可能な場合，距離の短い方を返す
                    if P_eval <= 0 and current_function <= 0:
                        if(l1 + l2 > l3 + l4):
                            return EdgeSet
                        else:
                            EdgeSet = copy.deepcopy(bestSet)

                    if(current_function < P_eval):
                        return EdgeSet
                    EdgeSet = copy.deepcopy(bestSet)

                elif current_function > P_eval:
                    EdgeSet = copy.deepcopy(bestSet)
                else:
                    P_eval = current_function
                    bestSet = copy.deepcopy(EdgeSet)
                    if first == 1:
                        return bestSet
            flag = False

        # # vとwが同じルートか判定
        # sameRoute = False
        # path = routeToPath(EdgeSet)
        # for r in path:
        #     node = np.unique(r)
        #     if v in node and w in node:
        #         sameRoute = True
        #
        # #ノードvとwが異なるルートに存在する時
        # if not(sameRoute):
            # """
            # 2-opt②
            # """
            # # print("2opt②やるよー")
            # l1 = cost[v_minus][v] # 元のエッジ
            # l2 = cost[w][w_plus] # 元のエッジ
            # l3 = cost[v][w]
            # l4 = cost[v_minus][w_plus]
            #
            # # if l1 + l2 > l3 + l4:
            # # print("2-opt② " + str(v) + ":" + str(w) + "適用")
            # try:
            #     EdgeSet.remove(link_v[0])
            #     EdgeSet.remove(link_w[1])
            #     EdgeSet.append([v, w])
            #     EdgeSet.append([v_minus, w_plus])
            #     flag = True
            # except ValueError:
            #     EdgeSet = copy.deepcopy(bestSet)
            #     flag = False
            #
            # if flag == True:
            #     for es in EdgeSet:
            #         if es[0] == es[1]:
            #             if es[0] == 0 and reduce_route == 1:
            #                 EdgeSet.remove([0, 0])
            #             else:
            #                 EdgeSet = copy.deepcopy(bestSet)
            #                 flag = False
            #
            # if flag == True:
            #     # デポを含まない巡回路ができた場合
            #     if(isHeiro(routeToPath(EdgeSet)) != 0):
            #         EdgeSet = copy.deepcopy(bestSet)
            #         flag = False
            #
            # if flag == True:
            #     current_function = penaltyFunction(EdgeSet, f_option)
            #     # ペナルティ関数により評価
            #     if(f_option == 2):
            #         # 元の解も改良解も実行可能な場合，距離の短い方を返す
            #         if P_eval <= 0 and current_function <= 0:
            #             if(l1 + l2 > l3 + l4):
            #                 return EdgeSet
            #             else:
            #                 EdgeSet = copy.deepcopy(bestSet)
            #
            #         if(current_function < P_eval):
            #             return EdgeSet
            #         EdgeSet = copy.deepcopy(bestSet)
            #     elif current_function > P_eval:
            #         EdgeSet = copy.deepcopy(bestSet)
            #     else:
            #         P_eval = current_function
            #         bestSet = copy.deepcopy(EdgeSet)
            #         return bestSet
            # flag = False
            # """
            # 2-opt③
            # """
            # # print("2opt③やるよー")
            # l1 = cost[v][v_plus] # 元のエッジ
            # l2 = cost[w_minus][w] # 元のエッジ
            # l3 = cost[v][w]
            # l4 = cost[v_plus][w_minus]
            #
            # # if l1 + l2 > l3 + l4:
            # # print("2-opt③ " + str(v) + ":" + str(w) + "適用")
            # try:
            #     EdgeSet.remove(link_v[1])
            #     EdgeSet.remove(link_w[0])
            #     EdgeSet.append([v, w])
            #     EdgeSet.append([v_plus, w_minus])
            #     flag = True
            # except ValueError:
            #     EdgeSet = copy.deepcopy(bestSet)
            #     flag = False
            #
            # if flag == True:
            #     # [n, n]のようなエッジを持たないように修正
            #     for es in EdgeSet:
            #         if es[0] == es[1]:
            #             if es[0] == 0 and reduce_route == 1:
            #                 EdgeSet.remove([0, 0])
            #             else:
            #                 EdgeSet = copy.deepcopy(bestSet)
            #                 flag = False
            # if flag == True:
            #     # デポを含まない巡回路ができた場合
            #     if(isHeiro(routeToPath(EdgeSet)) != 0):
            #         EdgeSet = copy.deepcopy(bestSet)
            #         flag = False
            #
            # if flag == True:
            #     current_function = penaltyFunction(EdgeSet, f_option)
            #     # ペナルティ関数により評価
            #     if(f_option == 2):
            #         # 元の解も改良解も実行可能な場合，距離の短い方を返す
            #         if P_eval <= 0 and current_function <= 0:
            #             if(l1 + l2 > l3 + l4):
            #                 return EdgeSet
            #             else:
            #                 EdgeSet = copy.deepcopy(bestSet)
            #
            #         if(current_function < P_eval):
            #             return EdgeSet
            #         EdgeSet = copy.deepcopy(bestSet)
            #
            #     elif current_function > P_eval:
            #         EdgeSet = copy.deepcopy(bestSet)
            #     else:
            #         P_eval = current_function
            #         bestSet = copy.deepcopy(EdgeSet)
            #         return bestSet
    return bestSet


"""
実行不可能解の修正操作
@INPUT:
    route: 解の2次元配列
@OUTPUT:
    modi_route: 修正後の解の2次元配列
"""
def modification(route):
    path = routeToPath(route)
    modi_route = []
    pena = False

    r, excess = checkCapacity(path)

    if r == []:
        return route, "succes"
    elif excess == False:
        print("修正不可")
        return route,"error"

    while(r):
        ExIdx = random.choice(r)
        # print("超過ルートのインデックス:{}".format(ExIdx))
        # print("実行不可能解あり")
        prePath = copy.deepcopy(path)

        for i in np.unique(path[ExIdx]):
            # print("i:{}".format(i))
            # print("np.unique(path[ExIdx]):{}".format(np.unique(path[ExIdx])))
            # sys.stdout.write("\r%d番目ルートの修正操作" % ExIdx)
            sys.stdout.flush()
            time.sleep(0.01)
            modi_route = Neighborhoods(i, path, f_option=2, reduce_route=0)

            path = routeToPath(modi_route)
            r, excess = checkCapacity(path)
            if(r == []):
                pena = True
                break
        # print("ルートに対する修正操作後の評価関数値:{}".format(penaltyFunction(modi_route, option=2)))
        if pena == True:
            break
        elif prePath == path:
            print("【modification】修正失敗")
            # print(route)
            return route,"error"

        r = []
        r, excess = checkCapacity(path)
    # print(modi_route)
    return modi_route,"success"

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
            I.append(int(i))
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
    E_A = P_A.getGenom() # 親Aのエッジリストを取得
    E_B = P_B.getGenom() # 親Bのエッジリストを取得

    """
    ステップ1: 親Aと親Bから集合G_ABを生成する
    """
    G_AB = [x for x in E_A + E_B if not (x in E_A and x in E_B)]

    # print("G_AB前:{}".format(G_AB))

    # G_ABから[0, 2]と[2, 0]のような同じエッジを示す要素を排除
    for e1, e2 in itertools.combinations(G_AB, 2):
        if e1[0] == e2[1] and e1[1] == e2[0]:
            G_AB.remove(e1)
            G_AB.remove(e2)
    # print("G_AB後:{}".format(G_AB))
    # graphPlot(E_A, isFirst=1, isLast=0, title="E_A")
    # graphPlot(E_B, isFirst=0, isLast=1, title="E_B")
    # graphPlot(G_AB, isFirst=0, isLast=1, title="G_AB")


    # 全く同じルートではG_ABが構築できない
    if G_AB == []:
        # print("親が同じルート構成")
        return False
    elif len(G_AB) % 2 == 1:
        print("G_AB内のエッジ数が奇数")
        return False
    # graphPlot(G_AB, isFirst=1, isLast=0, title="G_AB")

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

    if len(R_A) != len(R_B):
        print("G_ABが構成できない")
        return False

    error_flag = False
    P = [0]
    C = []
    while(len(G_AB)):
        # 次の閉路を探索するためにフラグを初期化
        ABflag = False
        # G_AB上のE_Aによるエッジが繋がるノードをランダムに選択する
        v_e = random.choice(np.unique(R_A))
        v_e_1 = v_e #最初の端点を保持する
        # print("【探索開始点】:{}".format(v_e))
        while (not(ABflag)):
            if(P[s] in R_B or s == 0):
                # 上で選択したノードv_eにつながるR_Aのエッジをeにセットする
                R_A = [x for x in R_A if (x and x in G_AB)]
                # print("R_A:{}".format(R_A))
                try:
                    e = random.choice(list(filter(lambda x: v_e in x, R_A)))
                except IndexError:
                    print("R_A内に探索中の点が見つかりません")
                    return False
                # print("e:{}".format(e))
                # G_ABから上で選択したエッジを取り除く
                G_AB = [x for x in G_AB if x != e]
                # print("eを除いたG_AB:{}".format(G_AB))
            else:
                R_B = [x for x in R_B if (x and x in G_AB)]
                # print("R_B:{}".format(R_B))
                try:
                    e = random.choice(list(filter(lambda x: v_e in x, R_B)))
                except IndexError:
                    print("R_B内に探索中の点が見つかりません")
                    error_flag = True
                    return False
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
    # print("AB-cycle:{}".format(C))
    if C == []:
        print("Cが空")
        return False
    return C


def edgeAssemblyCrossover(P_A, P_B, ABc):
    E_choice = 0.5
    E_A = P_A.getGenom() # 親Aのエッジリストを取得
    E_B = P_B.getGenom() # 親Bのエッジリストを取得
    E_set = []

    """
    ステップ3:E-setを構成する
    """
    E_set = random.choice(ABc) # Single戦略

    # while(True):
    #     for c in ABc:
    #         if E_choice > (random.randint(0, 100) / Decimal(100)):
    #             E_set += c[:]
    #     if(E_set != []):
    #         break

    # print("E-set:{}".format(E_set))
    # graphPlot(E_set, isFirst=0, isLast=1, title="E-set")

    """
    ステップ4:E-setを用いて中間個体を生成する
    """
    # E_AからE-setに含まれるE_Aに属する枝を取り除く
    interA = [x for x in E_A if not(x and x in E_set)]
    # E-setに含まれるE_Bに属する枝を付け加える
    interB = [x for x in E_B if (x and x in E_set)]
    intermediate = interA + interB

    # print("中間:{}".format(intermediate))

    """
    ステップ5:部分順回路が含まれる場合，結合する
    """

    subtour = isHeiro(routeToPath(intermediate))
    # print("閉路のインデックス{}".format(subtour))
    if(subtour != 0):
        # print("EAX Step5処理")
        child = EAXstep5(intermediate)
    else:
        child = intermediate
    # graphPlot(child, isFirst=0, isLast=1, title="child")
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
def EAXstep5(intermediate):
    # print("Step5")

    path = routeToPath(intermediate)
    subtourIndex = copy.deepcopy(isHeiro(path))
    # print(path)
    # print(subtourIndex)
    while(True):
        # graphPlot(intermediate, isFirst=0, isLast=0, title="Step5")
        # print("subtourIndex:{}".format(subtourIndex))
        best = float("inf")

        Ui = copy.deepcopy(path) # エッジ集合(3次元リスト)
        subnum = random.choice(subtourIndex) # サブツアーインデックスをランダムに選ぶ

        # print("Ui:{}".format(Ui))
        # for k in range(len(Ui)):
        #     print("U{}:{}".format(k, Ui[k]))
        # print("subnum:{}".format(subnum))
        # print("Ui:{}".format(Ui))
        # print("subtourIndex:{}".format(subtourIndex))
        # print("subnum:{}".format(subnum))
        Ur = Ui[int(subnum)]  # 選択したサブツアー
        Ui.pop(subnum)

        # print("Ui:{}".format(Ui))
        # print("Ur:{}".format(Ur))

        # UrとUiそれぞれのエッジの全ての組合せを調べる
        for e1, e2 in itertools.product(Ur, pathToRoute(Ui)):
            w1 = -cost[int(e1[0])][int(e1[1])] -cost[int(e2[0])][int(e2[1])] + \
            cost[int(e1[0])][int(e2[0])] + cost[int(e1[1])][int(e2[1])]

            w2 = -cost[int(e1[0])][int(e1[1])] -cost[int(e2[0])][int(e2[1])] + \
            cost[int(e1[0])][int(e2[1])] + cost[int(e1[1])][int(e2[0])]

            w = min(w1, w2)

            if w < best:
                best = w
                idx = 1 if w1 == min(w1, w2) else 2
                rme1 = e1
                rme2 = e2
                if(idx == 1):
                    adde1 = [int(e1[0]), int(e2[0])]
                    adde2 = [int(e1[1]), int(e2[1])]
                else:
                    adde1 = [int(e1[0]), int(e2[1])]
                    adde2 = [int(e1[1]), int(e2[0])]

        # 全ての組合せから-w(e)-w(e')+w(e")+w(e''')を最小にする
        # e∈Urとe∈Uj(j≠r)を探す
        intermediate.remove(rme1)
        intermediate.remove(rme2)
        intermediate.append(adde1)
        intermediate.append(adde2)

        path = routeToPath(intermediate)
        subtourIndex = isHeiro(path)

        if subtourIndex == 0:
            break

    return intermediate



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
    plt.savefig(saveDirectory + title +".png")  # save as png
    plt.pause(0.01)
    plt.clf()
    # plt.show()
    # return(0)

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
    # nx.draw_networkx_labels(G, pos, labels=labels, font_size=12) # デフォルト12
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
        # print("最初の経路:{}".format(penaltyFunction(edgeList, 0)))
        # print("ペナルティ関数値:{}".format(penaltyFunction(edgeList, 1)))
        # print("ルート数:{}".format(len(plot_path)))
        plt.title(title)
        plotDepot(title)
        # plt.pause(0.01)
        # plt.figure()


        plt.pause(0.01)
        plt.clf()

    # 連続プロット中
    if isLast == 0:
        plt.title(title)
        plt.pause(0.01)
        plt.clf()
    else:
        print(checkCapacity(routeToPath(edgeList)))
        plt.title(title)
        plotDepot(title)


if __name__ == '__main__':
    Capa = [160, 140, 200, 200, 200, 160, 140, 200, 200, 200, 200, 200, 200, 200]
    Vehicle = [5, 10, 8, 12, 17, 6, 11, 9, 14, 18, 7, 10, 11, 11]
    skip = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    S_OP = 3
    # 0: ランダム
    # 1: エッジ数
    # 2: 距離
    # 3: 需要量
    # 4: All
    if S_OP == 0:
        method = "ランダム"
    elif S_OP == 1:
        method = "エッジ数"
    elif S_OP == 2:
        method = "距離"
    elif S_OP == 3:
        method = "需要量"
    elif S_OP == 4:
        method = "All"


    for No in range(1, 15):
        if No in skip:
            continue


        filename = "vrpnc" + str(No)
        saveDirectory = "./output/MAdemand/5/"
        # 避難所情報のデータフレームを生成する
        # 引数[0]:ファイルパス，[1]:ファイル名
        df = createDataFrame("./csv/Christ/", filename)
        num_shelter = len(df.index)

        CAPACITY = Capa[No-1]
        m = Vehicle[No-1] + 3

        print("ファイル名:{}.csv".format(filename))
        print("繰り返す世代数:{}世代".format(MAX_GENERATION))
        print("集団数:{}個".format(MAX_GENOM_LIST))
        print("生成する子数:{}個".format(MAX_CHILDREN))
        print("トラック台数:{}台".format(m))
        print("トラック容量:{}".format(CAPACITY))
        # print("期間制約:{}".format(D))
        # print("α:{}".format(ALPHA))
        # print("β:{}".format(BETA))
        print("顧客数:{}".format(num_shelter-1))

        f = open(saveDirectory + "config/" + "Problem" + str(No) + "_config.csv", "w")
        writer = csv.writer(f, lineterminator="\n")
        paramArray = [["ファイル名", filename + ".csv"],
               ["顧客数", num_shelter-1],
               ["繰り返す世代数", MAX_GENERATION],
               ["集団数", MAX_GENOM_LIST],
               ["生成子数", MAX_CHILDREN],
               ["トラック台数(m+3)", m],
               ["トラック容量", CAPACITY],
               ["ルート分割方法", method]]

        writer.writerows(paramArray)
        paramArray = []

        result_df = pd.DataFrame(index=[], columns=['世代', 'ペナルティ関数値', '総移動コスト'])

        # 各避難所間の移動コスト行列を生成する
        # 2次元配列costで保持
        cost = createCostMatrix(num_shelter)
        # 第一世代の個体集団を生成
        current_generation_individual_group = []
        indi_count = 0

        start = time.time()
        m_time = 0
        print("現行の個体集団")
        while(True):
            # current_generation_individual_group.append(createGenom(num_shelter, m))
            """
            セービング法により初期解を生成
            """
            sa_route, distance = savingMethod(num_shelter, cost)
            # if len(sa_route) != m:
            #     continue
            path = savingRoute(sa_route) # ３次元解
            print("セービング法のルート数:{}".format(len(path)))
            route = pathToRoute(path) # ２次元解
            # graphPlot(route, isFirst=0, isLast=1, title="Saving Route")
            # sys.exit()
            path = routeSplit(path) # ルート数をmに固定
            # print("分割後ルート数:{}".format(len(path)))
            route = pathToRoute(path) # ２次元解
            print(checkCapacity(path))
            # graphPlot(route, isFirst=0, isLast=1, title="Split Route")

            # 局所探索用のランダムな並びを生成
            random_order = [i for i in range(1, num_shelter)]
            random.shuffle(random_order)
            # sys.exit()
            """
            局所探索
            """
            print("")
            # first改善山登り法による局所探索法によって解を改善
            for n, i in enumerate(random_order):
                prePath = copy.deepcopy(path)
                local_route = Local(i, path, f_option=1, reduce_route=0)
                path = routeToPath(local_route)
                if path == False:
                    path = copy.deepcopy(prePath)
                if path != prePath:
                    print("局所探索成功")
                    break
                # graphPlot(local_route, isFirst=0, isLast=0, title="local search")
                # sys.stdout.write("\r%d番目ノードの局所探索" % n)
                # sys.stdout.flush()
                # time.sleep(0.01)

            # route = pathToRoute(path)

            """
            修正操作
            """
            # if(penaltyFunction(route, 2) > 0):
            #     m_start = time.time()
            #     route,result = modification(route) #修正に失敗していたらresultにFalse
            #     m_time += time.time() - m_start
            #     if result == False:
            #         continue

            if len(path) != m:
                print("ルート数が異なる")
                continue
            # print("ここまである:{}".format(len(route)))
            ditance = penaltyFunction(route, option=0)
            evaluation = penaltyFunction(route, option=1)
            # sys.exit()
            # print(distance)
            # GAクラスに解，評価値，距離を保存
            current_generation_individual_group.append(ga.genom(route, evaluation, distance))

            print("個体【{}】:{}".format(indi_count, current_generation_individual_group[indi_count].getGenom()))
            print("評価{}".format(current_generation_individual_group[indi_count].getEvaluation()))
            print("距離{}".format(current_generation_individual_group[indi_count].getDistance()))
            print("-----------------------------------------------------------------")
            indi_count += 1
            sys.stdout.write("\r%d個初期解生成" % indi_count)
            sys.stdout.flush()
            time.sleep(0.01)
            if indi_count == MAX_GENOM_LIST:
                print("")
                break

        # graphPlot(current_generation_individual_group[0].getGenom(), isFirst=0, isLast=0, title="1st generation")
        monotonous = current_generation_individual_group[0].getDistance()
        mono_count = 0


        # 今世代の個体適用度を配列化する
        fits = [f.getEvaluation() for f in current_generation_individual_group]
        # print("fits:{}".format(fits))
        min_idx = fits.index(min(fits)) # 最小値を求める
        # print("min_idx:{}".format(min_idx))
        min_ = current_generation_individual_group[int(min_idx)].getDistance()
        min_Eval = current_generation_individual_group[int(min_idx)].getEvaluation()
        print("====第{}世代====".format(int(1)))
        print("最も優れた個体の評価値:{}".format(min_Eval))
        series = pd.Series([int(1), min_Eval, min_], index=result_df.columns)
        result_df = result_df.append(series, ignore_index = True)

        """
        ここまで第一世代
        この先繰り返し
        """
        for count_ in range(2, MAX_GENERATION + 1):
            #集団中の個体をランダムな順列に並べる
            order = setRondomOrder()

            for i in range(MAX_GENOM_LIST):
                # 集団中の全ての個体が丁度一度ずつ親P_Aとして選択される
                if i < MAX_GENOM_LIST -1:
                    P_A = current_generation_individual_group[order[i]]
                    P_B = current_generation_individual_group[order[i+1]]
                else:
                    P_A = current_generation_individual_group[order[i]]
                    P_B = current_generation_individual_group[order[0]]

                c = [] # 子解を代入するリスト
                c_cost = [] # 子の移動距離を代入するリスト
                c_eval = [] # 子のペナルティ関数値を代入するリスト
                """
                交叉：EAX
                """
                # print("P_A:{}".format(P_A.getGenom()))
                # print("P_B:{}".format(P_B.getGenom()))
                # graphPlot(P_A.getGenom(), isFirst=1, isLast=0, title="P_A")
                # graphPlot(P_B.getGenom(), isFirst=1, isLast=0, title="P_B")
                # EAXのステップ1~2を処理

                ABc = preEAX(P_A, P_B)
                # if ABc == False:
                #     print("False")

                # AB_cycleが構築できた場合，子解を生成する
                if ABc != False:
                    # EAXのステップ3~5
                    for j in range(MAX_CHILDREN):
                        # 局所探索用のランダムな並びを生成
                        random_order = [k for k in range(1, num_shelter)]
                        random.shuffle(random_order)

                        child = edgeAssemblyCrossover(P_A, P_B, ABc)
                        # route = child

                        # route,result = modification(child) #修正に失敗していたらresultにFalse
                        # if result == False:
                        #     child = copy.deepcopy(P_A.getGenom)

                        # sys.stdout.write("\r%d回目の交叉" % j)
                        # sys.stdout.flush()
                        # time.sleep(0.00001)


                        path = routeToPath(child)
                        """
                        EAX後の局所探索
                        f_option:
                            0→距離のみの評価
                            1→ペナルティ関数による評価
                            2→ペナルティ項が0以下になるように
                        """
                        local_count = 0
                        for n, eax in enumerate(random_order):
                            prePath = copy.deepcopy(path)
                            local_route = Local(eax, path, f_option=1, reduce_route=0)
                            path = routeToPath(local_route)
                            if path == False:
                                path = copy.deepcopy(prePath)
                            if path != prePath:
                                local_count += 1
                                if local_count >= 1:
                                    # print("局所探索成功")
                                    break


                        c.append(pathToRoute(path))
                        # c.append(route)
                        c_cost.append(penaltyFunction(c[j], option=0))
                        c_eval.append(penaltyFunction(c[j], option=1))

                    # """
                    # orderCrossoverにてプログラム全体の処理確認
                    # """
                    # 各両親に対してMAX_CHILDRENの数だけ子個体を生成する
                    # for j in range(MAX_CHILDREN):
                    #     c.append(orderCrossover(P_A, P_B))
                    #     a, b = createEdgeSet(c[j]) # bがコスト
                    #     Eval.append(penaltyFunction(a, option=1))
                    #     c_cost.append(b)

                    # 現在の親ペアから生成された子の中で一番コストの低い個体が親Aよりも
                    # 環境に適合している場合，親Aのインデックスを指定して入れ替える
                    if min(c_eval) < P_A.getEvaluation():
                        min_idx = c_eval.index(min(c_eval))
                        # print("子の評価値:{}".format(c_eval))
                        # print("min_idx:{}".format(min_idx))
                        # print("子の中で最小のコスト:{}".format(c_eval[min_idx]))
                        current_generation_individual_group[order[i]].setGenom(c[min_idx])
                        current_generation_individual_group[order[i]].setEvaluation(c_eval[min_idx])
                        current_generation_individual_group[order[i]].setDistance(c_cost[min_idx])
                        # print("入れ替わり後の親:{}".format(current_generation_individual_group[order[i]]\
                        #                            .getEvaluation()))

            # 今世代の個体適用度を配列化する
            fits = [f.getEvaluation() for f in current_generation_individual_group]
            # print("fits:{}".format(fits))
            min_idx = fits.index(min(fits)) # 最小値を求める
            # print("min_idx:{}".format(min_idx))
            min_ = current_generation_individual_group[int(min_idx)].getDistance()
            min_Eval = current_generation_individual_group[int(min_idx)].getEvaluation()

            if monotonous == min_Eval:
                mono_count += 1
            else:
                mono_count = 0
                monotonous = min_Eval


            # データフレームの行を世代によって更新
            series = pd.Series([int(count_), min_Eval, min_], index=result_df.columns)
            result_df = result_df.append(series, ignore_index = True)

            print("====第{}世代====".format(int(count_)))
            print("最も優れた個体の評価値:{}".format(min_Eval))

            # graphPlot(current_generation_individual_group[min_idx].getGenom(), \
            #           isFirst=0, isLast=0, title= str(count_) + "Generation")

            # 10世代以上適応度が変わらなかった場合
            if mono_count > 10:
                break

        elapsed_time = time.time() - start
        min_idx = fits.index(min(fits))
        # 最優個体の情報
        Best = current_generation_individual_group[min_idx]
        print("計算時間:{}".format(elapsed_time))
        print(result_df)
        print("最も優れた個体:{}".format(Best.getGenom()))
        print("最も優れた個体の移動距離:{}".format(Best.getDistance()))
        print("最も優れた個体の評価値:{}".format(Best.getEvaluation()))

        Bestpath = copy.deepcopy(routeToPath(Best.getGenom()))
        print(Bestpath)
        print("ルート数:{}".format(len(Bestpath)))

        paramArray = [["移動距離", Best.getDistance()],
               ["評価値", Best.getEvaluation()],
               ["ルート数", len(Bestpath)],
               ["計算時間", round(elapsed_time, 2)]]
        writer.writerows(paramArray)
        paramArray = []


        m_start = time.time()
        # first改善山登り法による局所探索法によって解を改善
        for n, i in enumerate(random_order):
            prePath = copy.deepcopy(Bestpath)
            local_route = Neighborhoods(i, Bestpath, f_option=0, reduce_route=0, first=0)
            Bestpath = routeToPath(local_route)
            if Bestpath == False:
                Bestpath = copy.deepcopy(prePath)
            # graphPlot(local_route, isFirst=0, isLast=0, title="local search")
        end_time = time.time() - m_start
        print("修正時間:{}".format(round(end_time, 2)))

        Bestroute = pathToRoute(Bestpath)
        r, excess = checkCapacity(Bestpath)
        print("ルート毎の需要:{}".format(excess))
        print("局所探索後")
        distance = penaltyFunction(Bestroute, 0)
        evaluation = penaltyFunction(Bestroute, 1)
        over = penaltyFunction(Bestroute, 2)

        print("距離:{}, 評価値:{}, 超過:{}".format(distance, evaluation, over))

        writer.writerow(["修正時間", round(end_time, 2)])
        writer.writerow(["ルート毎の需要", " "])
        writer.writerow(excess)
        writer.writerow(["局所探索後距離", distance])
        writer.writerow(["評価値", evaluation])
        writer.writerow(["超過", over])

        writer.writerow(["最優個体", " "])
        writer.writerow(Best.getGenom())
        writer.writerow(["最優ルート", " "])
        writer.writerows(Bestpath)

        result_df.to_csv(saveDirectory + "Problem" + str(No) + "_result.csv", index=False)
        graphPlot(current_generation_individual_group[min_idx].getGenom(), \
                  isFirst=1, isLast=0, title="Problem" + str(No) + " Last Generation")
        graphPlot(Bestroute, isFirst=1, isLast=0, title="Problem" + str(No) + " Best Route")
        f.close()
