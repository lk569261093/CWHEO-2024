import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import linregress
import matplotlib.pyplot as plt
import joblib
import spectral

# 读取Excel文件
file_path = '2.Zn光谱特征25.xlsx'
data = pd.read_excel(file_path)

# 提取土壤养分数据和光谱数据
y = data['Zn(mg/kg)']
X = data.drop(columns=['Zn(mg/kg)'])

# 使用指定的波长信息
selected_features = [2302.57, 2226.71, 2479.54, 2462.68, 1443.51, 420.27, 441.67]

# 确保在训练数据中排除没有对应波段索引的波段
valid_selected_features = [feature for feature in selected_features if feature in X.columns]

# 打印有效的选定特征
print(f"Valid selected features (bands): {valid_selected_features}")

# 提取有效的特征数据
X_selected = X[valid_selected_features]

# 数据预处理
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_selected)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# 保存用于标准化的 scaler_y 对象
joblib.dump(scaler_y, 'scaler_y.pkl')
joblib.dump(scaler_X, 'scaler_X.pkl')

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 定义随机森林模型并进行参数搜索
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2')
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

# 定义梯度提升回归模型并进行参数搜索
gbr = GradientBoostingRegressor(random_state=42)
param_grid_gbr = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search_gbr = GridSearchCV(gbr, param_grid_gbr, cv=5, scoring='r2')
grid_search_gbr.fit(X_train, y_train)
best_gbr = grid_search_gbr.best_estimator_

# 训练基础模型
best_rf.fit(X_train, y_train)
best_gbr.fit(X_train, y_train)

# 获取基础模型的预测
rf_train_pred = best_rf.predict(X_train)
gbr_train_pred = best_gbr.predict(X_train)
rf_test_pred = best_rf.predict(X_test)
gbr_test_pred = best_gbr.predict(X_test)

# 设置模型权重
rf_weight = 0.3
gbr_weight = 0.7

# 计算加权预测
y_train_pred_weighted = rf_weight * rf_train_pred + gbr_weight * gbr_train_pred
y_test_pred_weighted = rf_weight * rf_test_pred + gbr_weight * gbr_test_pred

# 评估模型
def evaluate_model(y_true, y_pred):
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    r2 = r_value**2
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# 评估加权模型
train_r2_weighted, train_rmse_weighted, train_mae_weighted = evaluate_model(y_train, y_train_pred_weighted)
test_r2_weighted, test_rmse_weighted, test_mae_weighted = evaluate_model(y_test, y_test_pred_weighted)

print(f'Training R2 (Weighted): {train_r2_weighted:.4f}, RMSE: {train_rmse_weighted:.4f}, MAE: {train_mae_weighted:.4f}')
print(f'Testing R2 (Weighted): {test_r2_weighted:.4f}, RMSE: {test_rmse_weighted:.4f}, MAE: {test_mae_weighted:.4f}')

# 逆标准化处理
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_train_pred_weighted_inv = scaler_y.inverse_transform(y_train_pred_weighted.reshape(-1, 1)).flatten()
y_test_pred_weighted_inv = scaler_y.inverse_transform(y_test_pred_weighted.reshape(-1, 1)).flatten()

# 保存结果到Excel文件
train_results = {
    'Actual Train': y_train_inv,
    'Predicted Train (Weighted)': y_train_pred_weighted_inv
}

train_results_df = pd.DataFrame(train_results)
train_results_df.to_excel('训练集实际值和预测值.xlsx', index=False)

test_results = {
    'Actual Test': y_test_inv,
    'Predicted Test (Weighted)': y_test_pred_weighted_inv
}

test_results_df = pd.DataFrame(test_results)
test_results_df.to_excel('测试集实际值和预测值.xlsx', index=False)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.plot(y_train_inv, label='Actual Train')
plt.plot(y_train_pred_weighted_inv, label='Predicted Train (Weighted)')
plt.legend()
plt.title('Training Set: Actual vs Predicted (Weighted)')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual Test')
plt.plot(y_test_pred_weighted_inv, label='Predicted Test (Weighted)')
plt.legend()
plt.title('Test Set: Actual vs Predicted (Weighted)')
plt.show()

# 读取高光谱图像数据
hdr_file = 'envi格式.hdr'  # 替换为实际的文件路径
dat_file = 'envi格式.dat'  # 替换为实际的文件路径
img = spectral.open_image(hdr_file)
data = img.load()

# 获取高光谱图像的波段数
num_bands = data.shape[2]

# 打印调试信息
print(f"Number of bands in hyperspectral image: {num_bands}")
print(f"Selected features (bands): {selected_features}")

# 从头文件中获取波长信息
wavelengths = np.array([
    390.320000, 394.600000, 398.880000, 403.160000, 407.440000, 411.710000,
    415.990000, 420.270000, 424.550000, 428.830000, 433.110000, 437.390000,
    441.670000, 445.950000, 450.230000, 454.500000, 458.780000, 463.060000,
    467.340000, 471.620000, 475.900000, 480.180000, 484.460000, 488.740000,
    493.010000, 497.290000, 501.570000, 505.850000, 510.130000, 514.410000,
    518.690000, 522.970000, 527.250000, 531.530000, 535.800000, 540.140000,
    544.390000, 548.660000, 552.910000, 557.170000, 561.460000, 565.740000,
    570.030000, 574.310000, 578.600000, 582.880000, 587.170000, 591.450000,
    595.740000, 600.030000, 604.310000, 608.600000, 612.880000, 617.170000,
    621.450000, 625.740000, 630.020000, 634.310000, 638.590000, 642.880000,
    647.070000, 651.300000, 655.540000, 659.820000, 664.110000, 668.440000,
    672.790000, 677.100000, 681.390000, 685.610000, 689.870000, 694.140000,
    698.360000, 702.580000, 706.870000, 711.150000, 715.420000, 719.700000,
    723.980000, 728.260000, 732.540000, 736.820000, 741.090000, 745.370000,
    749.650000, 753.930000, 758.200000, 762.480000, 766.760000, 771.040000,
    775.320000, 779.590000, 783.870000, 788.150000, 792.430000, 796.710000,
    800.980000, 805.260000, 809.540000, 813.820000,818.090000, 822.370000, 826.650000, 830.930000, 835.210000, 839.480000,
    843.760000, 848.040000, 852.320000, 856.600000, 860.880000, 865.150000,
    869.430000, 873.710000, 877.990000, 882.260000, 886.540000, 890.820000,
    895.100000, 899.380000, 903.650000, 907.930000, 912.210000, 916.490000,
    920.770000, 925.040000, 929.320000, 933.600000, 937.880000, 942.150000,
    946.430000, 950.710000, 954.990000, 959.270000, 963.540000, 967.820000,
    972.100000, 976.380000, 980.660000, 984.940000, 989.210000, 993.490000,
    997.950000, 1002.410000, 1004.770000, 1006.880000, 1011.340000, 1013.200000,
    1015.790000, 1020.260000, 1021.610000, 1024.710000, 1029.180000, 1030.050000,
    1038.470000, 1046.910000, 1055.320000, 1063.760000, 1072.180000, 1080.610000,
    1089.040000, 1097.460000, 1105.900000, 1114.310000, 1122.750000, 1131.180000,
    1139.600000, 1148.030000, 1156.450000, 1164.890000, 1173.310000, 1181.730000,
    1190.170000, 1198.590000, 1206.800000, 1215.190000, 1223.590000, 1232.330000,
    1240.760000, 1249.200000, 1257.660000, 1266.090000, 1274.550000, 1283.000000,
    1291.440000, 1299.900000, 1308.330000, 1316.790000, 1325.230000, 1333.680000,
    1342.140000, 1350.570000, 1359.030000, 1367.470000, 1375.930000, 1384.360000,
    1392.810000, 1401.270000, 1409.710000, 1418.170000, 1426.600000, 1435.060000,
    1443.510000, 1451.950000, 1460.400000, 1468.840000, 1477.300000, 1485.750000,
    1494.190000, 1502.640000, 1511.080000, 1519.540000, 1527.980000, 1536.430000,
    1544.880000, 1553.320000, 1560.930000, 1569.220000, 1577.600000, 1586.310000,
    1594.950000, 1603.380000, 1611.790000, 1620.200000, 1628.630000, 1637.050000,
    1645.460000, 1653.890000, 1662.310000, 1670.720000, 1679.150000, 1687.560000,
    1695.980000, 1704.410000, 1712.820000, 1721.250000, 1729.670000, 1738.070000,
    1746.500000, 1754.920000, 1763.330000, 1771.760000, 1780.180000, 1788.590000,
    1797.020000, 1805.430000, 1813.850000, 1822.280000, 1830.690000, 1839.110000,
    1847.540000, 1855.950000, 1864.380000, 1872.790000, 1881.200000, 1889.630000,
    1898.050000, 1906.460000, 1914.890000, 1923.310000, 1931.720000, 1940.150000,
    1948.560000, 1956.980000, 1965.410000, 1973.820000, 1982.250000, 1990.670000,
    1999.080000, 2007.500000, 2015.920000, 2024.330000, 2032.760000, 2041.180000,
    2049.600000, 2058.020000, 2066.430000, 2074.860000, 2083.280000, 2091.690000,
    2100.120000, 2108.540000, 2116.960000, 2125.400000, 2134.300000, 2142.310000,
    2150.870000, 2159.310000, 2167.720000, 2176.150000, 2184.590000, 2193.010000,
    2201.430000, 2209.870000, 2218.300000, 2226.710000, 2235.140000, 2243.580000,
    2252.010000, 2260.420000, 2268.860000, 2277.290000, 2285.700000, 2294.130000,
    2302.570000, 2311.000000, 2319.410000, 2327.850000, 2336.280000, 2344.700000,
    2353.120000, 2361.560000, 2369.980000, 2378.400000, 2386.840000, 2395.270000,
    2403.690000, 2412.120000, 2420.550000, 2428.970000, 2437.400000, 2445.830000,
    2454.260000, 2462.680000, 2471.110000, 2479.540000, 2487.960000, 2496.390000,
    2504.820000, 2513.250000
])

# 将波长值转换为波段索引
selected_bands = [np.where(wavelengths == band)[0][0] for band in valid_selected_features]

# 打印转换后的波段索引
print(f"Selected bands (indices): {selected_bands}")

# 检查是否有选中的波段
if not selected_bands:
    raise ValueError("No valid bands selected. Please check the selected features and the number of bands in the hyperspectral image.")

# 提取相应波段的数据
selected_data = data[:, :, selected_bands]

# 将数据转换为二维数组，每行代表一个像素，每列代表一个波段
n_rows, n_cols, n_bands = selected_data.shape
selected_data_reshaped = selected_data.reshape((n_rows * n_cols, n_bands))

# 加载用于标准化的 scaler 对象
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# 打印调试信息
print(f"Shape of selected data: {selected_data_reshaped.shape}")
print(f"Expected number of features: {scaler_X.n_features_in_}")

# 对数据进行标准化处理
selected_data_scaled = scaler_X.transform(selected_data_reshaped)

# 使用训练好的加权模型进行预测
rf_predictions = best_rf.predict(selected_data_scaled)
gbr_predictions = best_gbr.predict(selected_data_scaled)

# 计算加权预测
predictions_weighted = rf_weight * rf_predictions + gbr_weight * gbr_predictions

# 逆标准化处理
predictions_weighted_inv = scaler_y.inverse_transform(predictions_weighted.reshape(-1, 1)).flatten()

# 将预测结果转换为图像格式
predictions_image = predictions_weighted_inv.reshape((n_rows, n_cols))

# 保存预测结果为 ENVI 格式
output_hdr_file = 'predicted_zn_image.hdr'
output_dat_file = 'predicted_zn_image.dat'

# 创建 ENVI 头文件
hdr = spectral.envi.read_envi_header(hdr_file)
hdr['bands'] = 1
hdr['data type'] = 4  # 32-bit floating point
hdr['interleave'] = 'bsq'
hdr['byte order'] = 0
hdr['band names'] = ['Predicted Zn']

# 保存头文件
spectral.envi.write_envi_header(output_hdr_file, hdr)

# 保存数据文件
with open(output_dat_file, 'wb') as f:
    predictions_image.astype(np.float32).tofile(f)

print("预测结果已保存为 ENVI 格式的图像和头文件。")
