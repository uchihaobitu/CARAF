from CRVAE import CRVAE_demo
from PageRank import algo
import glob
from os.path import join, dirname, basename
import os
import torch
import argparse

# 设备配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResultAccumulator:
    """结果统计器（保持原始变量名）"""

    def __init__(self):
        # 原始版本计数器
        self.top1_cnt = 0
        self.top2_cnt = 0
        self.top3_cnt = 0
        self.top4_cnt = 0
        self.top5_cnt = 0
        self.total_cnt = 0

        # new版本计数器
        self.top1_cnt_new = 0
        self.top2_cnt_new = 0
        self.top3_cnt_new = 0
        self.top4_cnt_new = 0
        self.top5_cnt_new = 0
        self.total_cnt_new = 0

    def update(self, services, service_ranks, service_ranks_new):
        """更新统计结果"""
        # 原始版本统计
        if services in service_ranks[:1]:
            self.top1_cnt += 1
        if services in service_ranks[:2]:
            self.top2_cnt += 1
        if services in service_ranks[:3]:
            self.top3_cnt += 1
        if services in service_ranks[:4]:
            self.top4_cnt += 1
        if services in service_ranks[:5]:
            self.top5_cnt += 1
        self.total_cnt += 1

        # new版本统计
        if services in service_ranks_new[:1]:
            self.top1_cnt_new += 1
        if services in service_ranks_new[:2]:
            self.top2_cnt_new += 1
        if services in service_ranks_new[:3]:
            self.top3_cnt_new += 1
        if services in service_ranks_new[:4]:
            self.top4_cnt_new += 1
        if services in service_ranks_new[:5]:
            self.top5_cnt_new += 1
        self.total_cnt_new += 1

    def print_report(self):
        """打印结果报告（保持原始输出格式）"""
        # 不增加评分函数版本结果
        print(f"top1_accuracy:{self.top1_cnt / self.total_cnt}")
        print(f"top2_accuracy:{self.top2_cnt / self.total_cnt}")
        print(f"top3_accuracy:{self.top3_cnt / self.total_cnt}")
        print(f"top4_accuracy:{self.top4_cnt / self.total_cnt}")
        print(f"top5_accuracy:{self.top5_cnt / self.total_cnt}")
        avg5 = (self.top1_cnt + self.top2_cnt + self.top3_cnt + self.top4_cnt + self.top5_cnt) / (5 * self.total_cnt)
        print(f"Avg@5 Accuracy: {avg5}")

        # 增加评分函数版本结果
        print(f"top1_accuracy_new:{self.top1_cnt_new / self.total_cnt_new}")
        print(f"top2_accuracy_new:{self.top2_cnt_new / self.total_cnt_new}")
        print(f"top3_accuracy_new:{self.top3_cnt_new / self.total_cnt_new}")
        print(f"top4_accuracy_new:{self.top4_cnt_new / self.total_cnt_new}")
        print(f"top5_accuracy_new:{self.top5_cnt_new / self.total_cnt_new}")
        avg5_new = (self.top1_cnt_new + self.top2_cnt_new + self.top3_cnt_new +
                    self.top4_cnt_new + self.top5_cnt_new) / (5 * self.total_cnt_new)
        print(f"Avg@5 Accuracy: {avg5_new}")


def process_single_file(data_path, data_set1):
    """处理单个数据文件"""
    data_dir = dirname(data_path)
    services, metrics = basename(dirname(dirname(data_path))).split("_")
    services_number = basename(data_dir)

    # 创建输出目录
    output_dir = join(data_set1, f"{services}_{metrics}", services_number)
    os.makedirs(output_dir, exist_ok=True)

    # 训练模型
    CRVAE_demo.cause_trian(data_path, output_dir, device)

    # 构建文件路径
    base_filename = join(output_dir, f"{services}_{metrics}_{services_number}")
    csv_path = join(output_dir, 'notime_data.csv')
    graph_path = f"{base_filename}.npy"
    graph_path_new = f"{base_filename}new.npy"

    return services, metrics, csv_path, graph_path, graph_path_new


def run_pagerank_analysis(csv_path, graph_path, graph_path_new, services, metrics):
    """执行双版本PageRank分析"""
    # 不增加评分函数版本
    ranks = algo.algopage(csv_path, graph_path,
                          trigger_point=f"{services}_{metrics}",
                          root_cause=f"{services}_{metrics}")
    _service_ranks = [r.split("_")[0] for r in ranks]

    # 增加评分版本
    ranks_new = algo.algopage(csv_path, graph_path_new,
                              trigger_point=f"{services}_{metrics}",
                              root_cause=f"{services}_{metrics}")
    _service_ranks_new = [r.split("_")[0] for r in ranks_new]

    # 去重处理
    service_ranks = []
    for s in _service_ranks:
        if s not in service_ranks:
            service_ranks.append(s)

    service_ranks_new = []
    for s in _service_ranks_new:
        if s not in service_ranks_new:
            service_ranks_new.append(s)

    return service_ranks, service_ranks_new


def cause_nl(data_set1, ss_service_list, metric):
    """主流程"""
    accumulator = ResultAccumulator()

    pattern = join(data_set1, f"{ss_service_list[0]}_{metric[0]}", "**", "norm_data.npy")
    for data_path in glob.glob(pattern, recursive=True):
        try:
            # 处理单个文件
            services, metrics, csv_path, graph_path, graph_path_new = process_single_file(data_path, data_set1)

            # 运行分析
            service_ranks, service_ranks_new = run_pagerank_analysis(
                csv_path, graph_path, graph_path_new, services, metrics)

            # 更新统计
            accumulator.update(services, service_ranks, service_ranks_new)

        except Exception as e:
            print(f"处理 {data_path} 失败: {str(e)}")

    # 输出结果
    accumulator.print_report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set1', type=str, default="fse-ss")
    parser.add_argument('--ss_service_list', type=list,
                        default=['carts', 'catalogue', 'orders', 'payment', 'user'])

    parser.add_argument('--metric', type=list, default=['cpu', 'mem', 'delay', 'loss'])

    parser.add_argument('--num_workers', type=float, default=10)

    args = parser.parse_args()

    cause_nl(
        data_set1=args.data_set1,
        ss_service_list=args.ss_service_list,
        metric=args.metric
    )