#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Created by Kohei Kai(2017)
import networkx as nx
import matplotlib.pylab as plt

G = nx.Graph()
G.add_node("a")
G.add_nodes_from(["b","c"])
G.add_edge("a","c",weight=3)
G.add_edge("b","c",weight=5)

pos = nx.spring_layout(G)
edge_labels = {("a","c"):3,("b","c"):5}

nx.draw_networkx_nodes(G, pos, node_size=200, node_color="w")
nx.draw_networkx_edges(G, pos, width=1)
nx.draw_networkx_edge_labels(G, pos,edge_labels)
nx.draw_networkx_labels(G, pos ,font_size=16, font_color="r")

plt.xticks([])
plt.yticks([])
plt.show()
