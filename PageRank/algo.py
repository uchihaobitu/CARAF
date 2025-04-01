import argparse
import torch
import pandas as pd
import numpy as np
import json
import networkx as nx
# import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
# import os
# import sys
# import tqdm
from .models.utils import pearson_correlation, breaktie



def CreateGraph(edge, columns):
    G = nx.DiGraph()
    for c in columns:
        G.add_node(c)
    for pair in edge:
        p1,p2 = pair
        G.add_edge(columns[p2], columns[p1])
    return G

def get_edge_pair(npzfile):
    data = np.load(npzfile).T
    edge_pair = {}
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if data[i][j] != 0:
                edge_pair[(i,j)] = data[i,j]
    return edge_pair


def algopage(datafiles, graphfiles,trigger_point, root_cause):
    # edge_pair, columns = Run(datafiles)
    edge_pair = get_edge_pair(graphfiles)
    pruning = pd.read_csv(datafiles)
    columns = pruning.columns.tolist()

    print('edge_pair', edge_pair)
    G = CreateGraph(edge_pair, columns)

    #画图
    pos = nx.spring_layout(G, k=1.5, iterations=50)  # 增大 k 值和迭代次数
    # node_colors = ['red' if node == 'carts_cpu' else 'lightblue' for node in G.nodes()]
    node_colors = ['red' if node in [trigger_point, root_cause] else 'lightblue' for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_color=node_colors, font_weight='bold', node_size=700, font_size=9)

    plt.title(f"Graph Visualization for Sample ")
    # plt.figure(figsize=(100, 100))  # 增加图形尺寸
    plt.show()
    plt.close()  # 关闭图形，防止在内存中过多积累


    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)  # 尝试找到一个环
        if not cycle:
            break  # 如果没有环，退出循环

        edge_cor = []
        # 仅对环中的边进行操作
        for edge in tqdm(cycle, desc="Processing edges"):
            source, target = edge
            x = pruning[source].values  # Convert column to numpy array
            y = pruning[target].values
            edge_cor.append(pearson_correlation(x, y))
        print(edge_cor)

        # 使用torch对相关性进行排序
        tmp = torch.tensor(edge_cor)
        tmp_idx = torch.argsort(tmp)
        # 删除相关性最低的边，从而尝试破坏环
        source, target = cycle[tmp_idx[0]][0], cycle[tmp_idx[0]][1]
        G.remove_edge(source, target)

     # 生成并保存图形的可视化表示
    # plt.figure(figsize=(10, 8))
    node_colors = ['red' if node in [trigger_point, root_cause] else 'lightblue' for node in G.nodes()]

    nx.draw(G,pos, with_labels=True, node_color=node_colors, font_weight='bold', node_size=700, font_size=9)

    plt.title(f"Graph Visualization for Sample ")
    plt.figure(figsize=(100, 100))  # 增加图形尺寸
    plt.show()
    # plt.savefig(f"Graphv0.png")  # 保存图的可视化到文件
    plt.close()  # 关闭图形，防止在内存中过多积累
 
    dangling_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    personalization = {}
    for node in G.nodes():
        if node in dangling_nodes:
            personalization[node] = 1.0
        else:
            personalization[node] = 0.5
    pagerank = nx.pagerank(G, personalization=personalization)
    pagerank = dict(sorted(pagerank.items(), key=lambda x: x[1], reverse=True))

    pagerank = breaktie(pagerank, G, trigger_point)
    print(pagerank)
    return pagerank
