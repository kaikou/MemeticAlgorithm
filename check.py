#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)

import GeneticAlgorithm as ga
import numpy as np
import pandas as pd
import random
import data_frame as dfpy

MAX_GENOM_LIST = 1
VEHICLE = 3


if __name__ == '__main__':
    df = dfpy.createDataFrame("data_r101")
    print(df)
    num_shelter = 11

    dfpy.createGenom(num_shelter, 4)
    cost = dfpy.createCostMatrix(num_shelter, df)
    current_generation_individual_group = []
    print("現行の個体集団")
    for i in range(MAX_GENOM_LIST):
        current_generation_individual_group.append(dfpy.createGenom(num_shelter, VEHICLE))
        print(current_generation_individual_group[i].getGenom())

    for i in range(MAX_GENOM_LIST):
        evaluation_result = dfpy.evaluation(current_generation_individual_group[i], num_shelter, cost)
        current_generation_individual_group[i].setEvaluation(evaluation_result)
