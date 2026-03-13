import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import linregress
import matplotlib.pyplot as plt
import joblib

# 读取Excel文件
file_path = '2.SOM光谱特征25.xlsx'
data = pd.read_excel(file_path)

# 提取土壤养分数据和光谱数据
y = data['SOM(%)']
X = data.drop(columns=['SOM(%)'])

# 互信息法选择特征
mi = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns, name='互信息值')
mi_series.to_excel('互信息值.xlsx', index=True)

selected_features_mi = X.columns[mi > 0]
X_selected_mi = X[selected_features_mi]

# 数据预处理
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled_mi = scaler_X.fit_transform(X_selected_mi)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# 保存用于标准化的 scaler_y 对象
joblib.dump(scaler_y, 'scaler_y.pkl')

# 分割数据集
X_train_mi, X_test_mi, y_train, y_test = train_test_split(X_scaled_mi, y_scaled, test_size=0.2, random_state=42)

# 定义基础模型并进行参数搜索
gbr = GradientBoostingRegressor(random_state=42)
param_grid_gbr = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search_gbr = GridSearchCV(gbr, param_grid_gbr, cv=5, scoring='r2')
grid_search_gbr.fit(X_train_mi, y_train)
best_gbr = grid_search_gbr.best_estimator_

# 神经网络模型参数搜索
mlp = MLPRegressor(random_state=42, max_iter=50000)
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [0.0001, 0.001]
}
grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, cv=5, scoring='r2')
grid_search_mlp.fit(X_train_mi, y_train)
best_mlp = grid_search_mlp.best_estimator_

# 训练基础模型
best_gbr.fit(X_train_mi, y_train)
best_mlp.fit(X_train_mi, y_train)

# 获取基础模型的预测
gbr_train_pred = best_gbr.predict(X_train_mi)
mlp_train_pred = best_mlp.predict(X_train_mi)
gbr_test_pred = best_gbr.predict(X_test_mi)
mlp_test_pred = best_mlp.predict(X_test_mi)

# 设置模型权重
gbr_weight = 0.7
mlp_weight = 0.3

# 计算加权预测
y_train_pred_weighted = gbr_weight * gbr_train_pred + mlp_weight * mlp_train_pred
y_test_pred_weighted = gbr_weight * gbr_test_pred + mlp_weight * mlp_test_pred

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
