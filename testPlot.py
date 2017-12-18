#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created by Kohei Kai(2017)
"""
matplotlibでリアルタイムプロットする例

無限にsin関数をplotし続ける
"""
import math
import numpy as np
import matplotlib.pyplot as plt


"""sin波をリアルタイムプロットする関数"""
def waveplot(f, v):    # f:周波数[Hz]，v:電圧[％]
    pi = math.pi
    t = 1 / f
    plt.title('CurrentFrequency')
    plt.ylabel('VoltageRatio')
    plt.xlabel('Range:0-1 [s]')
    x = np.linspace(0, 2*pi, 1000)  # 0から2πまでの範囲を1000分割して0-1000で並べる
    plt.ylim(-100, 100)
    plt.xlim(0, 1000)
    y = v*(np.sin(x * f))

    plt.plot(y, color='limegreen')  # デフォルト(指定無し)だと水色
    plt.pause(0.01)  # 引数はsleep時間
    print("周波数", round(f, 3), "[Hz]・", "電圧", round(v, 2), "[％]・", "周期", round(t, 4), "[s]")

    # 関数の最後で消去しないとうまくプロットされない
    plt.cla()  # 現在描写されているグラフを消去

# 電圧・周波数の初期値
v = 0
f = 0

v1 = float(input("電圧変化率(100msでいくら増加するか)"))
f1 = float(input("周波数変化率(100msでいくら増加するか)"))
# plt.cla()で現在描写されているグラフを消去する
while 0 <= v < 100 :
    v += v1
    f += f1
    waveplot(f, v)

# while 0 <= v <= 101 :
#     v -= v1
#     f -= f1
#     waveplot(f, v)
