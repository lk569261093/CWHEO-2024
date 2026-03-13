import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matplotlib.patches import FancyArrowPatch

# 如果未安装 matplotlib_scalebar 库，请先安装
# 可以在命令行中运行：pip install matplotlib-scalebar
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# ==================== 新增导入 ====================
import matplotlib.ticker as ticker  # 用于自定义刻度
# ==================================================

# 定义文件名
img_filename = 'organic_matter_distribution.img'
hdr_filename = 'organic_matter_distribution.hdr'

# 检查文件是否存在
if not os.path.isfile(img_filename):
    print(f"Cannot find image file '{img_filename}'")
    exit()
if not os.path.isfile(hdr_filename):
    print(f"Cannot find header file '{hdr_filename}'")
    exit()

# 读取头文件信息
def read_envi_header(hdr_file):
    header = {}
    with open(hdr_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key_value = line.split('=')
                key = key_value[0].strip().lower()
                value = '='.join(key_value[1:]).strip()
                # 去除花括号{}
                value = value.strip('{}')
                header[key] = value
    return header

header = read_envi_header(hdr_filename)

# 提取图像参数
samples = int(header['samples'])  # 列数
lines = int(header['lines'])      # 行数
bands = int(header['bands'])      # 波段数
data_type = int(header['data type'])
interleave = header['interleave']
byte_order = int(header.get('byte order', '0'))  # 默认小端
header_offset = int(header.get('header offset', '0'))
data_ignore_value = float(header.get('data ignore value', '0.0'))

# 根据ENVI数据类型映射到numpy的数据类型
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

if data_type not in data_type_mapping:
    print(f"Unsupported data type '{data_type}'.")
    exit()

dtype = data_type_mapping[data_type]

# 设置字节序
if byte_order == 0:
    dtype = np.dtype(dtype).newbyteorder('<')  # 小端
else:
    dtype = np.dtype(dtype).newbyteorder('>')  # 大端

# 从头文件中读取地图信息以获取地理坐标
map_info_str = header.get('map info')
if map_info_str is None:
    print("Cannot find 'map info' in header.")
    exit()
else:
    # 解析 map info
    # 格式示例: {projection_name, x_ulc, y_ulc, lon_ulc, lat_ulc, x_pixel_size, y_pixel_size, ...}
    map_info = map_info_str.strip('{}').split(',')
    map_info = [item.strip() for item in map_info]
    if len(map_info) < 7:
        print("Incomplete 'map info' in header.")
        exit()
    projection_name = map_info[0]
    x_ulc = float(map_info[1])
    y_ulc = float(map_info[2])
    lon_ulc = float(map_info[3])
    lat_ulc = float(map_info[4])
    x_pixel_size = float(map_info[5])
    y_pixel_size = float(map_info[6])

# 读取图像数据
with open(img_filename, 'rb') as f:
    f.seek(header_offset)  # 如果有头偏移
    data = np.fromfile(f, dtype=dtype)

# 检查数据大小是否匹配
expected_size = lines * samples * bands
if data.size != expected_size:
    print(f"Data size does not match. Expected {expected_size}, got {data.size}.")
    exit()

# 重塑数据
if interleave.lower() == 'bip':
    data = data.reshape((lines, samples, bands))
elif interleave.lower() == 'bil':
    data = data.reshape((lines, bands, samples))
elif interleave.lower() == 'bsq':
    data = data.reshape((bands, lines, samples))
else:
    print(f"Unsupported interleave format '{interleave}'.")
    exit()

# 如果有多个波段，选择第一个波段进行展示和分析
if bands > 1:
    data = data[:, :, 0]

# 如果只有一个波段，简化数据结构
if bands == 1:
    data = data.reshape((lines, samples))

# 处理无效值
data = np.ma.masked_equal(data, data_ignore_value)

# 将小于等于 0 的值掩盖
data = np.ma.masked_less_equal(data, 0)

# 将数据中的最大值限制在 20
data = np.ma.clip(data, None, 20)

# 打印数据的最小值和最大值
print('Data min:', data.min())
print('Data max:', data.max())

# 生成经度和纬度数组
# 经度：从左到右递增
longitudes = lon_ulc + np.arange(samples) * x_pixel_size
# 纬度：从上到下递减
latitudes = lat_ulc - np.arange(lines) * y_pixel_size

# 设置 imshow 的范围 [left, right, bottom, top]
extent = [
    longitudes[0],           # left
    longitudes[-1],          # right
    latitudes[-1],           # bottom
    latitudes[0],            # top
]

# 计算图像的纵横比（高度/宽度）
data_height = abs(latitudes[0] - latitudes[-1])
data_width = abs(longitudes[-1] - longitudes[0])
aspect_ratio = data_height / data_width

# 设置图形大小，宽度固定，高度根据纵横比计算
fig_width = 8  # 英寸
fig_height = fig_width * aspect_ratio

# 创建图形和轴
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# 设置字体属性
font_properties = {'family': 'Times New Roman', 'size': 18}

# **创建自定义的离散颜色映射**
# 定义离散的颜色区间（5 到 20，每隔1一个颜色，共15个颜色）
bounds = np.arange(5, 21, 1)  # 5, 6, ..., 20
num_bins = len(bounds) - 1     # 15个区间

# 获取原始的 'viridis' 颜色映射，并指定离散的颜色数量
cmap = plt.get_cmap('tab20b', num_bins)  # 创建包含15个离散颜色的viridis颜色映射
cmap = cmap.copy()                        # 复制一份以便修改
cmap.set_bad(color='white')               # 设置被掩盖值的颜色为白色

# 定义颜色边界和规范化对象
norm = BoundaryNorm(bounds, cmap.N)

# 绘制数据，并使用自定义的离散颜色映射和规范化
im = ax.imshow(data, cmap=cmap, norm=norm, extent=extent, origin='upper', aspect='auto')

# 添加颜色条并设置标签
# 使用 inset_axes 创建嵌入式轴
cax = inset_axes(ax,
                width="3%",        # 颜色条的宽度
                height="50%",      # 颜色条的高度，占图像高度的50%
                loc='center right',
                bbox_to_anchor=(0.05, 0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0)

# 添加颜色条
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('SOM (g/kg)', fontdict=font_properties)
cbar.ax.tick_params(labelsize=18)
cbar.ax.yaxis.set_tick_params(labelright=True, labelleft=False)

# 使用格式化器来格式化刻度标签为整数
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

# 设置颜色条的刻度
cbar.set_ticks(bounds)
cbar.set_ticklabels([f"{tick:.0f}" for tick in bounds])  # 设置刻度标签为整数

# 设置标签和标题
ax.set_xlabel('Longitude', fontdict=font_properties)
ax.set_ylabel('Latitude', fontdict=font_properties)
ax.set_title('SOM Distribution Map', fontdict=font_properties)

# 设置刻度标签字体
ax.tick_params(axis='both', which='major', labelsize=18)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')

# 添加比例尺（可选）
# 如果需要添加比例尺，取消以下注释并确保计算正确
'''
lat_mean = (latitudes[0] + latitudes[-1]) / 2.0
meters_per_deg_lat = 111320  # 每度纬度对应的米数
meters_per_deg_lon = 111320 * math.cos(math.radians(lat_mean))  # 每度经度对应的米数

# 计算 x 和 y 方向每个像素对应的实际距离（米）
dx = x_pixel_size * meters_per_deg_lon
dy = y_pixel_size * meters_per_deg_lat

# 创建比例尺对象
scalebar = ScaleBar(dx, units="m", dimension='si-length', length_fraction=0.2,
                    location='lower left', border_pad=0.5, sep=5, box_alpha=0.5, 
                    font_properties={'family': 'Times New Roman', 'size': 14})
ax.add_artist(scalebar)
'''

# 调整图像布局
plt.tight_layout()

# 保存图像为高分辨率的 JPG 文件
output_filename = 'SOM_distribution.jpg'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

# ==================== 新增功能开始 ====================

# 计算统计信息
# 提取有效数据：去除掩盖的值
valid_data = data.compressed()

if valid_data.size == 0:
    print("No valid data available for statistics and histogram.")
    exit()

som_min = valid_data.min()
som_max = valid_data.max()
som_mean = valid_data.mean()
som_median = np.median(valid_data)

# 打印统计信息
print("SOM Concentration Statistics:")
print(f"Maximum Value: {som_max:.2f} g/kg")
print(f"Minimum Value: {som_min:.2f} g/kg")
print(f"Average Value: {som_mean:.2f} g/kg")
print(f"Median Value: {som_median:.2f} g/kg")

# 创建直方图
fig_hist, ax_hist = plt.subplots(figsize=(10, 6))

# 绘制直方图
n_bins = 50  # 根据需要调整分箱数量
ax_hist.hist(valid_data, bins=n_bins, color='skyblue', edgecolor='black')

# 设置直方图标题和标签
ax_hist.set_title('Histogram of SOM Concentration', fontsize=20, fontname='Times New Roman')
ax_hist.set_xlabel('SOM (g/kg)', fontsize=16, fontname='Times New Roman')
ax_hist.set_ylabel('Frequency', fontsize=16, fontname='Times New Roman')

# 设置字体
ax_hist.tick_params(axis='both', which='major', labelsize=14)
for label in ax_hist.get_xticklabels() + ax_hist.get_yticklabels():
    label.set_fontname('Times New Roman')

# 添加统计信息到直方图
text_str = (
    f"Max: {som_max:.2f} g/kg\n"
    f"Min: {som_min:.2f} g/kg\n"
    f"Mean: {som_mean:.2f} g/kg\n"
    f"Median: {som_median:.2f} g/kg"
)
# 将文本框放置在图表的右上角
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax_hist.text(0.98, 0.95, text_str, transform=ax_hist.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)

# 调整刻度标签字体
for label in ax_hist.get_xticklabels() + ax_hist.get_yticklabels():
    label.set_fontname('Times New Roman')

# 保存直方图
hist_output_filename = 'SOM_concentration_histogram.jpg'
plt.savefig(hist_output_filename, dpi=300, bbox_inches='tight')

# 显示直方图
plt.show()

# ==================== 新增功能结束 ====================
