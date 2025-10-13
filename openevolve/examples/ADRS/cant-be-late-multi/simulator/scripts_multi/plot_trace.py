import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_trace_data(filepath):
    """从JSON文件加载Trace数据，返回 (数据列表, 时间间隔小时数)。"""
    with open(filepath, 'r') as f:
        json_data = json.load(f)
    trace_list = [bool(x) for x in json_data['data']]
    gap_hours = json_data['metadata']['gap_seconds'] / 3600.0
    return trace_list, gap_hours

def plot_availability(ax, trace_data, gap_hours, title, color='dodgerblue'):
    """只绘制指定Trace的可用时段。"""
    # 绘制核心的可用性线条
    time_hours = [i * gap_hours for i in range(len(trace_data) + 1)]
    for i, is_unavailable in enumerate(trace_data):
        if not is_unavailable:
            ax.hlines(1, time_hours[i], time_hours[i+1], color=color, lw=5)

    # --- 修改部分 ---
    # 1. 使用标准的set_ylabel，并设置为水平显示(rotation=0)
    # 2. ha='right'让文字右对齐，va='center'让它垂直居中
    ax.set_ylabel(title, fontsize=11, rotation=0, ha='right', va='center')
    
    # 美化坐标轴
    ax.set_yticks([])
    ax.set_ylim(0, 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 将Y轴的轴线隐藏，因为我们用标签代替了
    ax.spines['left'].set_visible(False) 
    ax.spines['bottom'].set_position('zero')


# --- 主程序 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="分析并可视化两个Trace的重叠及统一可用性。")
    parser.add_argument('--trace-a', type=str, required=True, help='第一个Trace文件路径')
    parser.add_argument('--trace-b', type=str, required=True, help='第二个Trace文件路径')
    parser.add_argument('--output-file', type=str, default='availability_unified.png', help='输出图片路径')
    args = parser.parse_args()

    # 加载数据
    trace_a_data, gap_a = load_trace_data(args.trace_a)
    trace_b_data, gap_b = load_trace_data(args.trace_b)
    assert gap_a == gap_b, "两个Trace的gap_seconds必须相同才能进行分析！"

    # 数据处理与统计
    min_len = min(len(trace_a_data), len(trace_b_data))
    trace_a = np.array(trace_a_data[:min_len])
    trace_b = np.array(trace_b_data[:min_len])

    available_a = ~trace_a
    available_b = ~trace_b

    rate_a = np.mean(available_a) * 100
    rate_b = np.mean(available_b) * 100

    overlap_available = available_a & available_b
    overlap_rate = np.mean(overlap_available) * 100

    unified_available = available_a | available_b
    unified_rate = np.mean(unified_available) * 100
    
    unified_trace_data = ~unified_available

    print("\n--- 可用性统计分析 ---")
    print(f"Trace A 可用率: {rate_a:.2f}%")
    print(f"Trace B 可用率: {rate_b:.2f}%")
    print("-" * 25)
    print(f"重叠可用率 (A和B同时可用): {overlap_rate:.2f}%")
    print(f"统一可用率 (A或B至少一个可用): {unified_rate:.2f}%")
    print("--------------------------\n")

    # --- 修改部分 ---
    # 增加了画布高度 figsize=(16, 6)，并设置了子图间距 hspace
    fig, axes = plt.subplots(
        nrows=3, 
        ncols=1, 
        figsize=(16, 6), # <--- 修改：增加高度
        sharex=True,
        gridspec_kw={'hspace': 0.3} # <--- 修改：增加垂直间距
    )
    
    # 绘图
    plot_availability(axes[0], trace_a, gap_a, f"Trace A\n({rate_a:.1f}% Available)", 'dodgerblue')
    plot_availability(axes[1], trace_b, gap_b, f"Trace B\n({rate_b:.1f}% Available)", 'darkorange')
    plot_availability(axes[2], unified_trace_data, gap_a, f"Unified (A or B)\n({unified_rate:.1f}% Available)", 'mediumseagreen')

    # 设置全局格式
    fig.suptitle(f"Availability Analysis | Overlapping Rate: {overlap_rate:.2f}%", fontsize=16)
    axes[-1].set_xlabel("Time (hours)")
    
    # 使用 tight_layout 自动调整，现在效果会更好
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 调整布局以适应总标题
    
    # 保存并显示
    plt.savefig(args.output_file, dpi=200)
    print(f"图片已保存到: {args.output_file}")
    plt.show()