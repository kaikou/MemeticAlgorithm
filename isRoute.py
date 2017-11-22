#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)





def isRoute(edgeList, v_e):
    # エッジリストの長さが偶数かどうか
    if len(edgeList) % 2 == 0:
        if v_e in edgeList[0]:
            return 1
    else:
        return 0






if __name__ == '__main__':
    v_e = 0
    route = [[0, 8], [7, 8], [7, 10], [0, 10]]
    if(isRoute(route, v_e)):
        print("閉路")
    else:
        print("閉路じゃない")
