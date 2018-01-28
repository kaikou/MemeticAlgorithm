#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys

def Plot(filepath, filename):
    df = pd.read_csv("./output/" + filepath + filename + ".csv")
    # ax1 = plt.subplot()
    # ax2 = ax1.txinx()
    #
    # print(len(df))
    # plt.plot(range(len(df)), df["総移動コスト"], marker="o", label="cost")
    # plt.plot(range(len(df)), df["ペナルティ関数値"], marker="o", label="penalty")
    #
    # plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.legend()
    # plt.title(filename + ":result")
    # plt.xlabel("Generation")
    # plt.ylabel("Total Travel Cost")

    fig, ax = plt.subplots()  # create figure and axes object

    ax.plot(range(len(df)), df["総移動コスト"], label="cost", color="b")
    ax.set_ylabel("Cost")
    # ax.set_ylim(ymax=700)
    # ax1.set_yticks(np.arange(0, ))

    ax2 = ax.twinx()
    ax2.plot(range(len(df)), df["ペナルティ関数値"], label="penalty", color="g")
    ax2.set_ylabel("Evaluation")

    # ax.legend(loc='center left')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


    plt.xlim(xmax=len(df)-10)
    # plt.ylim(ymax=500)
    # ax.set_xlim(xmin=-5)
    ax.set_xlabel("Generation")


    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title(filename)
    plt.grid()
    # plt.legend()
    plt.subplots_adjust(hspace=0.7,bottom=0.2)
    plt.savefig("./output/" + filepath + "graph/" + filename + "_graph.png")  # save as png
    # plt.show()
    plt.pause(0.01)
    plt.clf()

if __name__ == '__main__':
    num = [1, 2, 3, 4, 5, 11, 12]
    filepath = "MAdemand/3/"

    for i in num:
        filename = "Problem" + str(i) + "_result"
        Plot(filepath, filename)
