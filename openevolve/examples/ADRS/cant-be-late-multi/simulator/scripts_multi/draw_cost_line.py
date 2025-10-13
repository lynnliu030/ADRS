import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from typing import List, Dict, Any

# 假设 parse_log_file 函数已存在
def parse_log_file(file_path: str) -> List[Dict[str, Any]]:
    """Parses the log file to extract strategy decision data."""
    parsed_data = []
    with open(file_path, 'r') as f:
        for line in f:
            if "STRATEGY_DECISION" in line:
                try:
                    json_str = line.split("STRATEGY_DECISION: ")[1]
                    data = json.loads(json_str)
                    parsed_data.append(data)
                except (IndexError, json.JSONDecodeError):
                    pass
    return parsed_data

def plot_full_analysis_with_progress_line(
    multi_data: List[Dict[str, Any]], 
    single_data: List[Dict[str, Any]], 
    total_work_needed: float, 
    deadline_seconds: float, 
    gap_seconds: float):
    """
    Plots a two-panel chart, with a unified progress line and conditional visibility.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(2, 1, figsize=(18, 15), sharex=True)

    # --- 数据准备 ---
    multi_ticks = np.array([d['tick'] for d in multi_data])
    multi_costs = np.array([d['accumulated_cost'] for d in multi_data])
    multi_work_cumulative = np.cumsum([d['last_tick_work'] for d in multi_data])

    single_ticks = np.array([d['tick'] for d in single_data])
    single_costs = np.array([d['accumulated_cost'] for d in single_data])
    single_work_cumulative = np.cumsum([d['last_tick_work'] for d in single_data])

    # ==================== 上图: 成本 vs. 时间 ====================
    ax[0].plot(single_ticks, single_costs, label='Single-Region Strategy', color='blue', linestyle='--', linewidth=2)
    ax[0].plot(multi_ticks, multi_costs, label='Multi-Region Strategy', color='red', linewidth=2.5)
    ax[0].set_title('Part 1: Accumulated Cost vs. Time', fontsize=16)
    ax[0].set_ylabel('Accumulated Cost ($)', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    formatter = mticker.FormatStrFormatter('$%1.2f')
    ax[0].yaxis.set_major_formatter(formatter)
    ax[0].legend(fontsize=12)

    # ==================== 下图: 进度 vs. 时间 (含条件显示) ====================
    # 1. 计算灰色“计划进度及格线”
    max_tick = max(multi_ticks.max(), single_ticks.max())
    unified_ticks = np.arange(0, max_tick + 1)
    # 匀速进度率 = 总工作量 / 总死线时间
    progress_rate = total_work_needed / deadline_seconds
    unified_progress = progress_rate * unified_ticks * gap_seconds
    ax[1].plot(unified_ticks, unified_progress, color='gray', linestyle=':', linewidth=2.5, label='On-Pace Progress (Required)')

    # 2. 条件化绘制：只在进度低于“及格线”时显示
    # 处理单区域策略 (蓝线)
    unified_progress_for_single = progress_rate * single_ticks * gap_seconds
    blue_masked_work = np.ma.masked_where(single_work_cumulative >= unified_progress_for_single, single_work_cumulative)
    ax[1].plot(single_ticks, blue_masked_work, label='Single-Region Progress (when behind)', color='blue', linestyle='--', linewidth=2)

    # 处理多区域策略 (红线)
    unified_progress_for_multi = progress_rate * multi_ticks * gap_seconds
    red_masked_work = np.ma.masked_where(multi_work_cumulative >= unified_progress_for_multi, multi_work_cumulative)
    ax[1].plot(multi_ticks, red_masked_work, label='Multi-Region Progress (when behind)', color='red', linewidth=2.5)

    # 3. 绘制总任务量完成线
    ax[1].axhline(y=total_work_needed, color='green', linestyle=':', linewidth=2, label=f'Total Work Required ({total_work_needed/3600:.1f}h)')

    # 格式化
    ax[1].set_title('Part 2: Effective Work Done vs. Time (Only showing when behind schedule)', fontsize=16)
    ax[1].set_xlabel('Time (Ticks)', fontsize=14)
    ax[1].set_ylabel('Cumulative Work Done (seconds)', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    # --- 统一格式和显示 ---
    fig.suptitle('Final Analysis: Cost vs. Efficiency Debt', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("final_analysis_chart.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # 请替换为你的日志文件路径
    MULTI_REGION_LOG_PATH = 'multi_region_log.log'
    SINGLE_REGION_A_LOG_PATH = 'single_region_A_log.log'
    
    # 任务参数 (根据你的实验配置)
    TOTAL_TASK_DURATION_SECONDS = 48 * 3600
    DEADLINE_SECONDS = 52 * 3600
    GAP_SECONDS = 600 # 每个tick是600秒
    
    multi_data = parse_log_file(MULTI_REGION_LOG_PATH)
    single_data = parse_log_file(SINGLE_REGION_A_LOG_PATH)
    
    if not multi_data or not single_data:
        print("无法从日志中解析足够的数据，请检查路径。")
    else:
        plot_full_analysis_with_progress_line(
            multi_data, 
            single_data,
            TOTAL_TASK_DURATION_SECONDS,
            DEADLINE_SECONDS,
            GAP_SECONDS
        )