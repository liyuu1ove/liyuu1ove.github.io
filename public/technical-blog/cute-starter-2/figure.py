import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. 创建画布
fig, ax = plt.subplots(figsize=(16, 3))
ax.set_xlim(-1, 25)
ax.set_ylim(-1, 2)
ax.axis('off')  # 隐藏自带的坐标轴

# 2. 构造颜色序列
# 原始布局 6:4 占领了 0, 4, 8, 12, 16, 20。我们用统一的深灰色表示它们。
# 剩下的空隙分为 4 组（由 4:1 控制），分别用 4 种不同的明亮颜色表示。
bg_color = '#7f7f7f'  # 6:4 占领的格子（基准墙）
gap_colors = ['#9999ff', '#ff9999', '#ffff99', '#99ff99']  # 4个空隙块的颜色

colors = [bg_color] * 24  # 先全部填满灰色

# 填充 4:1 的补集颜色
# 第 0 组空隙：Index 1, 2, 3
for j in [1,5,9,13,17,21]: colors[j] = gap_colors[0]
# 第 1 组空隙：Index 5, 6, 7
for j in [1,5,9,13,17,21]: colors[j+1] = gap_colors[1]
# 第 2 组空隙：Index 9, 10, 11
for j in [1,5,9,13,17,21]: colors[j+2] = gap_colors[2]


# 3. 绘制 24 个格子
box_y = 0.2
box_height = 0.6
box_width = 1.0

for i in range(24):
    # 绘制带黑边的彩色矩形
    rect = patches.Rectangle(
        (i, box_y), box_width, box_height, 
        edgecolor='black', facecolor=colors[i], linewidth=2
    )
    ax.add_patch(rect)
    
    # 在格子正中心写入 Index 数字
    ax.text(
        i + 0.5, box_y + box_height / 2, str(i), 
        ha='center', va='center', fontsize=16, fontname='DejaVu Sans'
    )

# 4. 绘制左侧的 "Index" 文本
ax.text(-0.5, box_y + box_height / 2, 'Index', ha='right', va='center', fontsize=18)

# 5. 绘制顶部的数学公式文本 (包含 LaTeX 格式)
formula_text = r"$\mathrm{Complement\ (6) : (4)\ under\ 24\ is\ (4) : (1)}$"
ax.text(
    0, 1.2, formula_text, 
    ha='left', va='bottom', fontsize=24, fontname='DejaVu Serif'
)

# 6. 调整布局并保存/显示
plt.tight_layout()
plt.savefig('complement_6_4.png', dpi=300, bbox_inches='tight')
plt.show()