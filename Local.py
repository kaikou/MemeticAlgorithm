#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)

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
from main import penaltyFunction

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
def Neighborhoods(v, path, f_option, reduce_route):
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
