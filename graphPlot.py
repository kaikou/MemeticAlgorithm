#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)

import pandas as pd
import matplotlib.pyplot as plt

filename = "data_r101"

df = pd.read_csv("./output/" + filename + "_result.csv")

# print(df["総移動コス"])
# print(len(df))
plt.plot(range(len(df)), df["総移動コスト"], marker="o", label="OX")
# plt.plot(range(len(df)), df["総移動コスト"], marker="o", label="OX")

plt.legend()
plt.title(filename + ":result")
plt.xlabel("Generation")
plt.ylabel("Total Travel Cost")

# plt.xlim(xmin=10)
# plt.ylim(ymax=500)

plt.savefig("./output/" + filename + "_result.png")  # save as png
plt.grid()
plt.show()
