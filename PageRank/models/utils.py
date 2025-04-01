import networkx as nx
import numpy as np
import torch

def pearson_correlation(x, y):
    """计算两个变量的皮尔逊相关系数（线性相关性度量）

        参数:
            x (array-like): 输入变量1，可以是list/numpy数组/tensor等
            y (array-like): 输入变量2，形状必须与x一致

        返回:
            float: 皮尔逊相关系数，范围[-1,1]，其中：
                  1 表示完全正相关
                  -1 表示完全负相关
                  0 表示无线性相关

        数学公式:
                  nΣxy - (Σx)(Σy)
            r = --------------------------
                  √[(nΣx²-(Σx)²)(nΣy²-(Σy)²)]

        """

    if len(x) != len(y):
        raise ValueError("The lengths of the input variables must be the same.")
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    n = len(x)

    sum_x = torch.sum(x)
    sum_y = torch.sum(y)
    sum_xy = torch.sum(x * y)
    sum_x_sq = torch.sum(x ** 2)
    sum_y_sq = torch.sum(y ** 2)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = torch.sqrt((n * sum_x_sq - sum_x ** 2) * (n * sum_y_sq - sum_y ** 2))

    if denominator == 0:
        return 0

    correlation = numerator / denominator
    return correlation.item()



def breaktie(pagerank, G, trigger_point):
    """基于网络距离的PageRank分数平局打破机制

        参数:
            pagerank (dict): 节点PageRank分数字典，格式为 {节点: 分数}
            G (nx.Graph): 网络图对象，需支持最短路径计算
            trigger_point (str): 触发节点名称，若为"None"则直接返回原排序

        返回:
            list: 重新排序后的节点列表，排序规则为：
                  1. 首先按PageRank分数降序
                  2. 分数相同的节点按到trigger_point的最短距离降序排列

        处理逻辑:
            1. 遍历PageRank排序结果
            2. 当遇到分数相同的节点组时：
               - 计算组内每个节点到trigger_point的最短路径距离
               - 按距离降序重新排列该组节点
            3. 无路径可达的节点距离记为0
        """

    if trigger_point == "None":
        return pagerank
    
    rank = []
    tmp_rank = []
    last_score = 0    
    for cnt, (node, score) in enumerate(pagerank.items()):
        if last_score != score:
            if len(tmp_rank) == 0:
                last_score = score
                rank.append(node)
            else:
                ad = []
                for i in range(len(tmp_rank)):
                    try: 
                        distance = nx.shortest_path_length(G, source=trigger_point, target=node)
                    except nx.NetworkXNoPath:
                        distance = 0
                    ad.append(distance)
                ad = np.array(ad)
                # dis_rank = np.argsort(ad, reverse=True)
                dis_rank = np.argsort(ad)[::-1]
                for i in range(len(dis_rank)):
                    rank.append(tmp_rank[dis_rank[i]])
                tmp_rank = [node]
        else:
            tmp_rank.append(node)
            if cnt == len(pagerank)-1:
                ad = []
                for i in range(len(tmp_rank)):
                    try: 
                        distance = nx.shortest_path_length(G, source=trigger_point, target=node)
                    except nx.NetworkXNoPath:
                        distance = 0
                    ad.append(distance)
                ad = np.array(ad)
                # dis_rank = np.argsort(ad, reverse=True)
                dis_rank = np.argsort(ad)[::-1]
                for i in range(len(dis_rank)):
                    rank.append(tmp_rank[dis_rank[i]])
    return rank
    
