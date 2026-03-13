import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # **新增**

# **设置全局字体为 Times New Roman**
plt.rcParams['font.family'] = 'Times New Roman'

# 定义文件名
img_filename = 'predicted_zn_image.dat'
hdr_filename = 'predicted_zn_image.hdr'

# 检查文件是否存在
if not os.path.isfile(img_filename):
    print(f"Cannot find image file '{img_filename}'")
    exit()
if not os.path.isfile(hdr_filename):
    print(f"Cannot find header file '{hdr_filename}'")
    exit()

# 读取 ENVI 头文件的函数
def read_envi_header(hdr_file):
    header = {}
    in_wavelength = False
    wavelength_data = ''
    with open(hdr_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith(';'):
                continue  # 跳过空行和注释
            if '=' in line:
                key_value = line.split('=', 1)
                key = key_value[0].strip().lower()
                value = key_value[1].strip()
                if '{' in value and '}' not in value:
                    # 多行条目的开始
                    in_wavelength = True
                    value = value.strip('{')
                    wavelength_data += value + ' '
                else:
                    # 单行条目
                    value = value.strip('{}')
                    header[key] = value
            else:
                # 多行条目的继续
                if in_wavelength:
                    if '}' in line:
                        line = line.strip('}')
                        in_wavelength = False
                    wavelength_data += line + ' '
        if wavelength_data:
            header['wavelength'] = wavelength_data.strip()
    return header

header = read_envi_header(hdr_filename)

# 提取图像参数
samples = int(header['samples'])  # 列数
lines = int(header['lines'])      # 行数
bands = int(header['bands'])      # 波段数
data_type = int(header['data type'])
interleave = header['interleave']
byte_order = int(header['byte order'])
header_offset = int(header.get('header offset', '0'))
data_ignore_value = float(header.get('data ignore value', '0.0'))

# ENVI 数据类型映射到 numpy 数据类型
data_type_mapping = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.complex64,
    9: np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}

dtype = data_type_mapping[data_type]

# 设置字节序
if byte_order == 0:
    dtype = np.dtype(dtype).newbyteorder('<')  # 小端
else:
    dtype = np.dtype(dtype).newbyteorder('>')  # 大端

# 读取图像数据
with open(img_filename, 'rb') as f:
    f.seek(header_offset)  # 如果有头偏移
    data = np.fromfile(f, dtype=dtype)

# 检查数据大小是否匹配
expected_size = lines * samples * bands
if data.size != expected_size:
    print(f"Data size does not match. Expected {expected_size}, got {data.size}.")
    exit()

# 根据 interleave 格式重塑数据
if interleave.lower() == 'bip':
    data = data.reshape((lines, samples, bands))
elif interleave.lower() == 'bil':
    data = data.reshape((lines, bands, samples))
elif interleave.lower() == 'bsq':
    data = data.reshape((bands, lines, samples))
else:
    print(f"Unsupported interleave format '{interleave}'.")
    exit()

# 如果只有一个波段，简化数据结构
if bands == 1:
    data = data.reshape((lines, samples))

# 处理无效值
data = np.ma.masked_equal(data, data_ignore_value)

# 处理数据
# 将小于等于 0 的值掩盖
data = np.ma.masked_less_equal(data, 0)

# 如果需要限制数据的最大值，可以取消注释以下行并设置 max_value
# max_value = 100  # 设置所需的最大值
# data = np.ma.clip(data, None, max_value)

# 创建自定义的颜色映射（可选）
cmap = plt.get_cmap('tab20b').copy()
cmap.set_bad(color='white')  # 将被掩盖的值设置为白色
#cmap = plt.get_cmap('plasma') #含盐量
#cmap = plt.get_cmap('viridis')#环境有机质
'''
常用 Colormap：
'viridis'：Matplotlib 的默认颜色映射，蓝绿到黄。
'plasma'：紫红到黄。
'inferno'：深紫到黄。
'magma'：深紫到浅黄。
'cividis'：蓝绿到黄绿，色盲友好型。
优先选择感知均匀的颜色映射：
推荐使用 'viridis'： 默认且适用于许多情况，颜色从深紫到黄绿，视觉感知均匀。
其他选项： 'plasma'（更温暖的颜色）、'inferno'（高对比度）、'magma'（暗色调）。
根据数据特点选择：
顺序数据： 使用顺序颜色映射，如 'viridis'、'plasma'。
有中心值的数据： 使用梯度颜色映射，如 'coolwarm'，强调数据偏离中心值的程度。
分类数据： 使用离散颜色映射，如 'tab10'、'Set3'。
考虑色盲友好性：
使用对色盲用户友好的颜色映射，如 'cividis'、'viridis'。
'''
# 设置图形和轴
fig, ax = plt.subplots(figsize=(10, 8))

# 显示图像
im = ax.imshow(data, cmap=cmap, origin='upper', aspect='auto')

# **调整颜色条的大小和位置**
cax = inset_axes(ax,
                 width="3%",       # 颜色条的宽度
                 height="50%",     # 颜色条的高度（图像高度的 50%）
                 loc='center right',
                 bbox_to_anchor=(0.05, 0, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0)

# **在指定的轴上添加颜色条**
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Zn Concentration (mg/Kg)', fontsize=16)

# **设置颜色条刻度标签的字体**
cbar.ax.yaxis.set_tick_params(labelsize=14)
for t in cbar.ax.get_yticklabels():
    t.set_fontname('Times New Roman')

# 标注轴
ax.set_xlabel('Column Number', fontsize=18)
ax.set_ylabel('Row Number', fontsize=18)
ax.set_title('Zn Distribution Map', fontsize=20)

# 添加行列号作为刻度标签
# 设置合理的刻度数量以避免过度拥挤
num_col_ticks = min(samples, 10)  # 根据需要调整
num_row_ticks = min(lines, 10)    # 根据需要调整
ax.set_xticks(np.linspace(0, samples - 1, num_col_ticks, dtype=int))
ax.set_yticks(np.linspace(0, lines - 1, num_row_ticks, dtype=int))

# 将刻度标签设置为行列号
ax.set_xticklabels(np.linspace(1, samples, num_col_ticks, dtype=int))
ax.set_yticklabels(np.linspace(1, lines, num_row_ticks, dtype=int))

# 调整刻度标签的大小和字体
ax.tick_params(axis='both', which='major', labelsize=12)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')

# 保存图像
plt.savefig('zn_distribution_map.jpg', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
