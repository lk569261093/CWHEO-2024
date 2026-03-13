import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import linregress
import matplotlib.pyplot as plt
import joblib
import spectral
from spectral import envi
import re
import warnings

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# 设置中文字体，避免字体缺失警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取Excel文件
file_path = '2.含盐量光谱特征25.xlsx'
data = pd.read_excel(file_path)

# 提取目标变量和特征
y = data['含盐量(g/kg)']
X = data.drop(columns=['含盐量(g/kg)'])

# 互信息法选择特征
mi = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns, name='互信息值')
mi_series.to_excel('互信息值.xlsx', index=True)

# 选择互信息值大于0.2的特征
threshold = 0.2
selected_features_mi = X.columns[mi > threshold]
X_selected_mi = X[selected_features_mi]

# 输出参与建模的波段
selected_features_mi_df = pd.DataFrame(selected_features_mi, columns=['波段'])
selected_features_mi_df.to_excel('参与建模的波段.xlsx', index=False)

# 打印有效的选定特征
print(f"Valid selected features (波段): {selected_features_mi.tolist()}")

# 检查 y 的统计信息
print("\n原始 y 数据统计：")
print(y.describe())

# 数据预处理
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 拟合并转换特征和目标
X_scaled_mi = scaler_X.fit_transform(X_selected_mi)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

print("\n标准化后的 y 数据统计：")
print(pd.Series(y_scaled).describe())

# 保存用于标准化的 scaler 对象
joblib.dump(scaler_y, 'scaler_y.pkl')
joblib.dump(scaler_X, 'scaler_X.pkl')

# 分割数据集
X_train_mi, X_test_mi, y_train, y_test = train_test_split(
    X_scaled_mi, y_scaled, test_size=0.2, random_state=42
)

# 定义基础模型
rf = RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)
svr = SVR(C=40, epsilon=0.01)

# 训练基础模型
rf.fit(X_train_mi, y_train)
svr.fit(X_train_mi, y_train)

# 定义元模型
meta_model = Ridge(alpha=6)

# 创建堆叠回归器
stacking_regressor = StackingRegressor(
    estimators=[('rf', rf), ('svr', svr)],
    final_estimator=meta_model,
    cv=5
)

# 训练堆叠模型
stacking_regressor.fit(X_train_mi, y_train)

# 保存堆叠模型
joblib.dump(stacking_regressor, '环境监测含盐量最优模型.pkl')

# 预测
y_train_pred_stacking = stacking_regressor.predict(X_train_mi)
y_test_pred_stacking = stacking_regressor.predict(X_test_mi)

# 评估模型
def evaluate_model(y_true, y_pred, dataset_name=''):
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    r2 = r_value ** 2
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{dataset_name} R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return r2, rmse, mae

print("\n===== 模型评估（未裁剪） =====")
train_r2_stacking, train_rmse_stacking, train_mae_stacking = evaluate_model(
    y_train, y_train_pred_stacking, 'Training Set (Stacking)'
)
test_r2_stacking, test_rmse_stacking, test_mae_stacking = evaluate_model(
    y_test, y_test_pred_stacking, 'Testing Set (Stacking)'
)

# 逆标准化处理
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_train_pred_stacking_inv = scaler_y.inverse_transform(
    y_train_pred_stacking.reshape(-1, 1)
).flatten()
y_test_pred_stacking_inv = scaler_y.inverse_transform(
    y_test_pred_stacking.reshape(-1, 1)
).flatten()

# 添加统计信息打印
print("\n逆标准化后的预测 y 数据统计（训练集）：")
print(pd.Series(y_train_pred_stacking_inv).describe())
print("\n逆标准化后的预测 y 数据统计（测试集）：")
print(pd.Series(y_test_pred_stacking_inv).describe())

# 将负值设为 0
y_train_pred_stacking_inv_clipped = np.clip(y_train_pred_stacking_inv, a_min=0, a_max=None)
y_test_pred_stacking_inv_clipped = np.clip(y_test_pred_stacking_inv, a_min=0, a_max=None)

# 评估裁剪后的模型
print("\n===== 模型评估（裁剪后） =====")
train_r2_clipped, train_rmse_clipped, train_mae_clipped = evaluate_model(
    y_train_inv, y_train_pred_stacking_inv_clipped, 'Training Set (Stacking & Clipped)'
)
test_r2_clipped, test_rmse_clipped, test_mae_clipped = evaluate_model(
    y_test_inv, y_test_pred_stacking_inv_clipped, 'Testing Set (Stacking & Clipped)'
)

# 保存结果到Excel文件
train_results = {
    'Actual Train': y_train_inv,
    'Predicted Train (Stacking & Clipped)': y_train_pred_stacking_inv_clipped
}

train_results_df = pd.DataFrame(train_results)
train_results_df.to_excel('训练集实际值和预测值.xlsx', index=False)

test_results = {
    'Actual Test': y_test_inv,
    'Predicted Test (Stacking & Clipped)': y_test_pred_stacking_inv_clipped
}

test_results_df = pd.DataFrame(test_results)
test_results_df.to_excel('测试集实际值和预测值.xlsx', index=False)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(y_train_inv, label='Actual Train')
plt.plot(y_train_pred_stacking_inv_clipped, label='Predicted Train (Stacking & Clipped)')
plt.legend()
plt.title('训练集：实际值 vs 预测值 (堆叠 & 裁剪)')
plt.xlabel('样本索引')
plt.ylabel('含盐量 (g/kg)')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Test')
plt.plot(y_test_pred_stacking_inv_clipped, label='Predicted Test (Stacking & Clipped)')
plt.legend()
plt.title('测试集：实际值 vs 预测值 (堆叠 & 裁剪)')
plt.xlabel('样本索引')
plt.ylabel('含盐量 (g/kg)')
plt.show()

# 读取高光谱图像数据
hdr_file = 'Imagedata_R.hdr'  # 替换为你的实际头文件路径
img = spectral.open_image(hdr_file)
img_data = img.load()

# 获取高光谱图像的维度
n_rows, n_cols, n_bands = img_data.shape

# 获取头文件的元数据
metadata = img.metadata

# 尝试获取 no-data 值
nodata_value = None
if 'data ignore value' in metadata:
    nodata_value = float(metadata['data ignore value'])
elif 'no data' in metadata:
    nodata_value = float(metadata['no data'])
elif 'data ignore value 1' in metadata:
    nodata_value = float(metadata['data ignore value 1'])  # 某些文件可能分波段定义
else:
    print("No no-data value found in metadata. 请检查头文件并手动设置无效值。默认设置为 -9999.")
    nodata_value = -9999  # 默认值，可以根据实际情况调整

# 获取选定的波段编号，并转换为整数
try:
    band_numbers = [int(re.search(r'\d+', band).group()) for band in selected_features_mi]
except AttributeError as e:
    raise ValueError("波段名称格式不正确，应包含数字部分，例如 'B34'。") from e

# 转换为零基索引
band_indices = [band - 1 for band in band_numbers]

# 检查波段编号是否在有效范围内
max_band_number = n_bands
invalid_bands = [band for band in band_numbers if band < 1 or band > max_band_number]
if invalid_bands:
    print(f"Warning: 以下波段编号超出高光谱图像的范围（1-{max_band_number}）：{invalid_bands}")
    # 移除这些波段：
    band_indices = [idx for idx, band in zip(band_indices, band_numbers) if 1 <= band <= max_band_number]
    if not band_indices:
        raise ValueError("所有选定的波段编号均无效。")

print(f"Selected bands (indices): {band_indices}")

# 检查是否有选中的波段
if not band_indices:
    raise ValueError("No valid bands selected. 请检查选择的特征和高光谱图像中的波段数量。")

# 提取相应波段的数据
selected_data = img_data[:, :, band_indices]

# 将数据转换为二维数组，每行代表一个像素，每列代表一个波段
n_selected_bands = selected_data.shape[2]
selected_data_reshaped = selected_data.reshape((n_rows * n_cols, n_selected_bands))

# 创建无效像素的掩膜
if nodata_value is not None:
    # 假设无效像素在所有选定波段都有无效值
    mask = np.all(selected_data_reshaped == nodata_value, axis=1)
else:
    # 如果没有无效值定义，则全部像素认为有效
    mask = np.zeros(selected_data_reshaped.shape[0], dtype=bool)

# 创建有效像素掩膜
valid_pixels_mask = ~mask

print(f"Total pixels: {selected_data_reshaped.shape[0]}")
print(f"Valid pixels: {np.sum(valid_pixels_mask)}")
print(f"Invalid pixels: {np.sum(mask)}")

# 加载用于标准化的 scaler 对象
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# 打印调试信息
print(f"Shape of selected data: {selected_data_reshaped.shape}")
print(f"Expected number of features: {scaler_X.n_features_in_}")

# 检查特征数量是否匹配
if selected_data_reshaped.shape[1] != scaler_X.n_features_in_:
    raise ValueError(f"Feature mismatch: selected data has {selected_data_reshaped.shape[1]} features, but scaler expects {scaler_X.n_features_in_} features.")

# 提取有效像素的数据
selected_data_valid = selected_data_reshaped[valid_pixels_mask]

# 对有效像素进行标准化处理
try:
    img_scaled_valid = scaler_X.transform(selected_data_valid)
except Exception as e:
    raise ValueError(f"标准化时出错: {e}")

# 加载训练好的堆叠模型
stacking_regressor = joblib.load('环境监测含盐量最优模型.pkl')

# 使用训练好的堆叠模型进行预测
y_pred_scaled_valid = stacking_regressor.predict(img_scaled_valid)

# 逆标准化预测结果
y_pred_valid = scaler_y.inverse_transform(y_pred_scaled_valid.reshape(-1, 1)).flatten()

# 将负值设为 0
y_pred_valid_clipped = np.clip(y_pred_valid, a_min=0, a_max=None)

# 创建一个包含缺失值的预测数组
nodata_value_output = -9999  # 可以根据需要设置为其他值
y_pred_img = np.full((n_rows * n_cols,), nodata_value_output, dtype=np.float32)

# 将预测结果赋值给有效像素
y_pred_img[valid_pixels_mask] = y_pred_valid_clipped

# 将预测结果转换为图像格式
y_pred_img = y_pred_img.reshape((n_rows, n_cols))

# 更新头文件信息
metadata_out = metadata.copy()
metadata_out['lines'] = n_rows
metadata_out['samples'] = n_cols
metadata_out['bands'] = 1  # 只有一个预测结果波段
metadata_out['data type'] = 4  # float32
metadata_out['interleave'] = 'bsq'
metadata_out['byte order'] = 0
metadata_out['band names'] = ['Predicted Salinity']
metadata_out['no data'] = nodata_value_output  # 设置缺失值

# 保存预测结果为 ENVI 格式
output_file = '含盐量_distribution'
envi.save_image(f'{output_file}.hdr', y_pred_img, metadata=metadata_out, force=True)

print("预测结果已保存为 ENVI 格式的图像和头文件。")

# 可视化预测结果
plt.figure(figsize=(12, 8))
# 使用掩膜在可视化时忽略缺失值
y_pred_img_masked = np.ma.masked_equal(y_pred_img, nodata_value_output)
plt.imshow(y_pred_img_masked, cmap='viridis')
cbar = plt.colorbar(label='含盐量 (g/kg)')
plt.title('预测的含盐量分布图')
plt.xlabel('样本列')
plt.ylabel('样本行')
plt.show()
