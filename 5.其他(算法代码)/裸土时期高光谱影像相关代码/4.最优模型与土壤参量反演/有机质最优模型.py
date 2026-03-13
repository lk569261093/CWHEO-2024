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

# 读取Excel文件
file_path = '2.有机质光谱特征25.xlsx'
data = pd.read_excel(file_path)

# 提取土壤养分数据和光谱数据
y = data['有机质(g/kg)']
X = data.drop(columns=['有机质(g/kg)'])

# 互信息法选择特征
mi = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns, name='互信息值')
mi_series.to_excel('互信息值.xlsx', index=True)

selected_features_mi = X.columns[mi > 0.3]
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

# 定义和训练基础模型
rf = RandomForestRegressor(max_depth=8, n_estimators=50, random_state=42)    
svr = SVR(C=10, epsilon=0.01)

rf.fit(X_train_mi, y_train)
svr.fit(X_train_mi, y_train)

# 定义元模型
meta_model = Ridge(alpha=5)

# 创建堆叠回归器
stacking_regressor = StackingRegressor(
    estimators=[('rf', rf), ('svr', svr)],
    final_estimator=meta_model,
    cv=5
)

# 训练堆叠模型
stacking_regressor.fit(X_train_mi, y_train)

# 保存模型
joblib.dump(stacking_regressor, '环境监测有机质最优模型.pkl')

# 预测
y_train_pred_stacking = stacking_regressor.predict(X_train_mi)
y_test_pred_stacking = stacking_regressor.predict(X_test_mi)

# 评估模型
def evaluate_model(y_true, y_pred):
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    r2 = r_value**2
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# 评估堆叠模型
train_r2_stacking, train_rmse_stacking, train_mae_stacking = evaluate_model(y_train, y_train_pred_stacking)
test_r2_stacking, test_rmse_stacking, test_mae_stacking = evaluate_model(y_test, y_test_pred_stacking)

print(f'Training R2 (Stacking): {train_r2_stacking:.4f}, RMSE: {train_rmse_stacking:.4f}, MAE: {train_mae_stacking:.4f}')
print(f'Testing R2 (Stacking): {test_r2_stacking:.4f}, RMSE: {test_rmse_stacking:.4f}, MAE: {test_mae_stacking:.4f}')

# 逆标准化处理
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_train_pred_stacking_inv = scaler_y.inverse_transform(y_train_pred_stacking.reshape(-1, 1)).flatten()
y_test_pred_stacking_inv = scaler_y.inverse_transform(y_test_pred_stacking.reshape(-1, 1)).flatten()

# 保存结果到Excel文件
train_results = {
    'Actual Train': y_train_inv,
    'Predicted Train (Stacking)': y_train_pred_stacking_inv
}

train_results_df = pd.DataFrame(train_results)
train_results_df.to_excel('训练集实际值和预测值.xlsx', index=False)

test_results = {
    'Actual Test': y_test_inv,
    'Predicted Test (Stacking)': y_test_pred_stacking_inv
}

test_results_df = pd.DataFrame(test_results)
test_results_df.to_excel('测试集实际值和预测值.xlsx', index=False)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.plot(y_train_inv, label='Actual Train')
plt.plot(y_train_pred_stacking_inv, label='Predicted Train (Stacking)')
plt.legend()
plt.title('Training Set: Actual vs Predicted (Stacking)')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual Test')
plt.plot(y_test_pred_stacking_inv, label='Predicted Test (Stacking)')
plt.legend()
plt.title('Test Set: Actual vs Predicted (Stacking)')
plt.show()
