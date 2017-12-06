#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ベンチマークのtxtファイルをcsvファイルに変換するプログラム
# Created by Kohei Kai(2017)

import csv



def txt2csv(filepath, filename):
    f = open("./csv/" + filename + ".csv", "w")
    writer = csv.writer(f, lineterminator="\n")

    lineCount = 0
    data = []
    csv_ = []
    num = ""
    flag = 0
    readfile = filepath + filename + ".txt"
    for line in open(readfile, "r"):
        lineCount += 1
        column = 0
        if lineCount < 10:
            continue
        for char in line: # 1文字ずつ
            if column >= 4:
                continue
            if char == " " or char == "\n":
                if flag == 0:
                    continue
                else:
                    # if column == 0:
                    #     column += 1
                    #     data = []
                    #     continue
                    data.append(int(num))
                    column += 1
                    num = ""
                    flag = 0
                    if column == 4:
                        csv_.append(data)
                        data = []
                    continue
            num = num + char
            # print(int(num))
            flag = 1
    csv_.pop()
    print(csv_)
    writer.writerows(csv_)
    f.close()



if __name__ == '__main__':
    txt2csv("./data/solomon_100/", "101")
