#!/usr/bin/env python3
"""Figure 5.6(a)(b) — PPO智能体训练奖励曲线（分开输出）

读取 ppo_results.json 分别输出 W1 和 W2 两张图。
每张图三条线：基准线（灰色虚线）、训练奖励（深蓝色实线）、测试奖励（红橙色实线）。
完整框线，无最终标注，中英文文字清晰阅读。

用法: python gen_fig5_6.py
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── 字体 ──────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 字号（9~10.5pt 清晰阅读）
TITLE_SIZE = 11
AXIS_LABEL_SIZE = 10.5
TICK_SIZE = 9
LEGEND_SIZE = 9

# ── 路径 ──────────────────────────────────────
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_PATH = os.path.join(PKG, 'output/chapter5/results/ppo_results.json')
FIG_OUT = os.path.join(PKG, 'output/chapter5')

FALLBACK = os.path.join(PKG, 'experiments', 'chapter5', 'results', 'ppo_results.json')

fp = JSON_PATH if os.path.exists(JSON_PATH) else FALLBACK
with open(fp) as f:
    data = json.load(f)

w1 = data['w1']
w2 = data['w2']

# ── 配色 ──────────────────────────────────────
C_DARK_BLUE = '#4A7FB5'     # 训练奖励（标准蓝）
C_RED_ORANGE = '#D9543A'    # 测试奖励（红橙）
C_GRAY = '#999999'           # 基准线（灰）
C_GRID = '#D6E4F0'           # 网格（浅蓝）

# ── 公共设置 ──────────────────────────────────
def make_plot(title, winfo, train_color, outpath, figsize=(7, 5)):
    """绘制单张PPO奖励曲线"""
    train_hist = winfo['train_rewards_history']
    test_hist = winfo['test_rewards_history']
    baseline = winfo['baseline_reward']
    ep_train = np.arange(1, len(train_hist) + 1)
    ep_test = np.arange(1, len(test_hist) + 1)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor('white')

    # 三条曲线：基线（最底下）→ 测试 → 训练（最顶上）
    ax.axhline(y=baseline, color=C_GRAY, linestyle='--', linewidth=1.2, zorder=2,
               label=f'静态基线={baseline:.1f}')
    ax.plot(ep_test, test_hist, color=C_RED_ORANGE, linewidth=1.5, linestyle='-',
            label='测试奖励 (test)', zorder=4)
    ax.plot(ep_train, train_hist, color=train_color, linewidth=1.8,
            label='训练奖励 (train)', zorder=5)

    # 轴标签
    ax.set_xlabel('训练轮数 (episode)', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('累积奖励', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=10)

    # 完整框线
    for spine_name in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color('#333333')
        ax.spines[spine_name].set_linewidth(0.8)

    # 刻度字号
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    # 浅色网格
    ax.grid(True, color=C_GRID, linestyle='-', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # 图例半透明，不遮挡曲线
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right',
              framealpha=0.55, edgecolor='#cccccc')

    # y轴：基于数据范围加12% padding，不贴边
    all_y = train_hist + test_hist + [baseline]
    y_min, y_max_base = min(all_y), max(all_y)
    y_range = y_max_base - y_min
    padding = y_range * 0.12
    ax.set_ylim(y_min - padding, y_max_base + padding)

    plt.tight_layout(pad=1.2)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'✅ saved: {outpath}')


# ── W1（现输出）──
make_plot(
    title='W1 (1-6月训练, 7-8月测试)',
    winfo=w1,
    train_color=C_DARK_BLUE,
    outpath=os.path.join(FIG_OUT, 'fig5_6a_ppo_reward_w1.png')
)

# ── W2（后输出）──
make_plot(
    title='W2 (1-9月训练, 10-12月测试)',
    winfo=w2,
    train_color=C_DARK_BLUE,
    outpath=os.path.join(FIG_OUT, 'fig5_6b_ppo_reward_w2.png')
)

print('✅ fig5_6a(w1) + fig5_6b(w2) 完成')
