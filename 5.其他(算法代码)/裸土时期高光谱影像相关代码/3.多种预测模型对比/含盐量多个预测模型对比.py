import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import linregress
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = '2.含盐量光谱特征25.xlsx'
data = pd.read_excel(file_path)

# 提取土壤养分数据和光谱数据
y = data['含盐量(g/kg)']
X = data.drop(columns=['含盐量(g/kg)'])

# 互信息法选择特征
mi = mutual_info_regression(X, y)
mi_series = pd.Series(mi, index=X.columns, name='互信息值')
mi_series.to_excel('互信息值.xlsx', index=True)

selected_features_mi = X.columns[mi > 0.2]
X_selected_mi = X[selected_features_mi]

# 递归特征消除法选择特征
ridge_for_rfe = Ridge()
rfe = RFE(estimator=ridge_for_rfe, n_features_to_select=10)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
X_selected_rfe = X[selected_features_rfe]

# 数据预处理
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled_mi = scaler_X.fit_transform(X_selected_mi)
X_scaled_rfe = scaler_X.fit_transform(X_selected_rfe)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# 分割数据集
X_train_mi, X_test_mi, y_train, y_test = train_test_split(X_scaled_mi, y_scaled, test_size=0.2, random_state=42)
X_train_rfe, X_test_rfe = train_test_split(X_scaled_rfe, test_size=0.2, random_state=42)

# 定义参数网格（岭回归）
param_grid_ridge = {
    'alpha': [0.1, 1, 10, 100]  # 正则化强度，值越大正则化越强
}

# 初始化岭回归模型
ridge = Ridge()

# 使用网格搜索和交叉验证选择最佳参数（互信息法）
grid_search_ridge_mi = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='r2')
grid_search_ridge_mi.fit(X_train_mi, y_train)

# 输出最佳参数（互信息法）
print(f'Best alpha (Ridge, MI): {grid_search_ridge_mi.best_params_["alpha"]}')

# 使用最佳参数重新训练模型（互信息法）
best_ridge_mi = grid_search_ridge_mi.best_estimator_
best_ridge_mi.fit(X_train_mi, y_train)

# 预测（互信息法）
y_train_pred_ridge_mi = best_ridge_mi.predict(X_train_mi)
y_test_pred_ridge_mi = best_ridge_mi.predict(X_test_mi)

# 使用网格搜索和交叉验证选择最佳参数（递归特征消除法）
grid_search_ridge_rfe = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='r2')
grid_search_ridge_rfe.fit(X_train_rfe, y_train)

# 输出最佳参数（递归特征消除法）
print(f'Best alpha (Ridge, RFE): {grid_search_ridge_rfe.best_params_["alpha"]}')

# 使用最佳参数重新训练模型（递归特征消除法）
best_ridge_rfe = grid_search_ridge_rfe.best_estimator_
best_ridge_rfe.fit(X_train_rfe, y_train)

# 预测（递归特征消除法）
y_train_pred_ridge_rfe = best_ridge_rfe.predict(X_train_rfe)
y_test_pred_ridge_rfe = best_ridge_rfe.predict(X_test_rfe)

# 定义参数网格（随机森林）
param_grid_rf = {
    'n_estimators': [50, 100, 200],  # 树的数量
    'max_depth': [None, 10, 20, 30]  # 树的最大深度
}

# 初始化随机森林模型
rf = RandomForestRegressor(random_state=42)

# 使用网格搜索和交叉验证选择最佳参数（互信息法）
grid_search_rf_mi = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2')
grid_search_rf_mi.fit(X_train_mi, y_train)

# 输出最佳参数（互信息法）
print(f'Best params (RF, MI): {grid_search_rf_mi.best_params_}')

# 使用最佳参数重新训练模型（互信息法）
best_rf_mi = grid_search_rf_mi.best_estimator_
best_rf_mi.fit(X_train_mi, y_train)

# 预测（互信息法）
y_train_pred_rf_mi = best_rf_mi.predict(X_train_mi)
y_test_pred_rf_mi = best_rf_mi.predict(X_test_mi)

# 使用网格搜索和交叉验证选择最佳参数（递归特征消除法）
grid_search_rf_rfe = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2')
grid_search_rf_rfe.fit(X_train_rfe, y_train)

# 输出最佳参数（递归特征消除法）
print(f'Best params (RF, RFE): {grid_search_rf_rfe.best_params_}')

# 使用最佳参数重新训练模型（递归特征消除法）
best_rf_rfe = grid_search_rf_rfe.best_estimator_
best_rf_rfe.fit(X_train_rfe, y_train)

# 预测（递归特征消除法）
y_train_pred_rf_rfe = best_rf_rfe.predict(X_train_rfe)
y_test_pred_rf_rfe = best_rf_rfe.predict(X_test_rfe)

# 线性回归模型
linear_reg = LinearRegression()
linear_reg.fit(X_train_mi, y_train)
y_train_pred_linear = linear_reg.predict(X_train_mi)
y_test_pred_linear = linear_reg.predict(X_test_mi)

# 主成分回归模型
#pca = PCA(n_components=10)  # 主成分数量
#X_train_pca = pca.fit_transform(X_train_mi)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_mi)
X_test_pca = pca.transform(X_test_mi)
linear_reg_pca = LinearRegression()
linear_reg_pca.fit(X_train_pca, y_train)
y_train_pred_pca = linear_reg_pca.predict(X_train_pca)
y_test_pred_pca = linear_reg_pca.predict(X_test_pca)

# 偏最小二乘回归模型
#pls = PLSRegression(n_components=10)  # 主成分数量
#pls.fit(X_train_mi, y_train)

# 调整 n_components 参数
n_components = min(X_train_mi.shape[0], X_train_mi.shape[1]) - 1
pls = PLSRegression(n_components=n_components)
pls.fit(X_train_mi, y_train)
y_train_pred_pls = pls.predict(X_train_mi).flatten()
y_test_pred_pls = pls.predict(X_test_mi).flatten()

# 支持向量机回归模型
svr = SVR()
param_grid_svr = {
    'C': [0.1, 1, 10],  # 惩罚参数，值越大对误差的惩罚越大
    'epsilon': [0.01, 0.1, 1]  # epsilon-SVR中的epsilon参数，控制模型的精度
}
grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='r2')
grid_search_svr.fit(X_train_mi, y_train)
best_svr = grid_search_svr.best_estimator_
best_svr.fit(X_train_mi, y_train)
y_train_pred_svr = best_svr.predict(X_train_mi)
y_test_pred_svr = best_svr.predict(X_test_mi)

# 梯度提升回归模型
gbr = GradientBoostingRegressor(random_state=42)
param_grid_gbr = {
    'n_estimators': [50, 100, 200],  # 弱学习器的数量
    'learning_rate': [0.01, 0.1, 0.2]  # 学习率，控制每个弱学习器对最终模型的贡献
}
grid_search_gbr = GridSearchCV(gbr, param_grid_gbr, cv=5, scoring='r2')
grid_search_gbr.fit(X_train_mi, y_train)
best_gbr = grid_search_gbr.best_estimator_
best_gbr.fit(X_train_mi, y_train)
y_train_pred_gbr = best_gbr.predict(X_train_mi)
y_test_pred_gbr = best_gbr.predict(X_test_mi)

# 神经网络模型
mlp = MLPRegressor(random_state=42, max_iter=50000)  # 设置最大迭代次数为50000
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # 隐藏层的结构
    'activation': ['relu', 'tanh'],  # 激活函数
    'solver': ['adam', 'lbfgs'],  # 优化算法
    'alpha': [0.0001, 0.001]  # L2正则化参数
}
grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, cv=5, scoring='r2')
grid_search_mlp.fit(X_train_mi, y_train)
best_mlp = grid_search_mlp.best_estimator_
best_mlp.fit(X_train_mi, y_train)
y_train_pred_mlp = best_mlp.predict(X_train_mi)
y_test_pred_mlp = best_mlp.predict(X_test_mi)

# 评估模型
def evaluate_model(y_true, y_pred):
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    r2 = r_value**2
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# 岭回归评估
train_r2_ridge_mi, train_rmse_ridge_mi, train_mae_ridge_mi = evaluate_model(y_train, y_train_pred_ridge_mi)
test_r2_ridge_mi, test_rmse_ridge_mi, test_mae_ridge_mi = evaluate_model(y_test, y_test_pred_ridge_mi)

train_r2_ridge_rfe, train_rmse_ridge_rfe, train_mae_ridge_rfe = evaluate_model(y_train, y_train_pred_ridge_rfe)
test_r2_ridge_rfe, test_rmse_ridge_rfe, test_mae_ridge_rfe = evaluate_model(y_test, y_test_pred_ridge_rfe)

# 随机森林评估
train_r2_rf_mi, train_rmse_rf_mi, train_mae_rf_mi = evaluate_model(y_train, y_train_pred_rf_mi)
test_r2_rf_mi, test_rmse_rf_mi, test_mae_rf_mi = evaluate_model(y_test, y_test_pred_rf_mi)

train_r2_rf_rfe, train_rmse_rf_rfe, train_mae_rf_rfe = evaluate_model(y_train, y_train_pred_rf_rfe)
test_r2_rf_rfe, test_rmse_rf_rfe, test_mae_rf_rfe = evaluate_model(y_test, y_test_pred_rf_rfe)

# 线性回归评估
train_r2_linear, train_rmse_linear, train_mae_linear = evaluate_model(y_train, y_train_pred_linear)
test_r2_linear, test_rmse_linear, test_mae_linear = evaluate_model(y_test, y_test_pred_linear)

# 主成分回归评估
train_r2_pca, train_rmse_pca, train_mae_pca = evaluate_model(y_train, y_train_pred_pca)
test_r2_pca, test_rmse_pca, test_mae_pca = evaluate_model(y_test, y_test_pred_pca)

# 偏最小二乘回归评估
train_r2_pls, train_rmse_pls, train_mae_pls = evaluate_model(y_train, y_train_pred_pls)
test_r2_pls, test_rmse_pls, test_mae_pls = evaluate_model(y_test, y_test_pred_pls)

# 支持向量机回归评估
train_r2_svr, train_rmse_svr, train_mae_svr = evaluate_model(y_train, y_train_pred_svr)
test_r2_svr, test_rmse_svr, test_mae_svr = evaluate_model(y_test, y_test_pred_svr)

# 梯度提升回归评估
train_r2_gbr, train_rmse_gbr, train_mae_gbr = evaluate_model(y_train, y_train_pred_gbr)
test_r2_gbr, test_rmse_gbr, test_mae_gbr = evaluate_model(y_test, y_test_pred_gbr)

# 神经网络评估
train_r2_mlp, train_rmse_mlp, train_mae_mlp = evaluate_model(y_train, y_train_pred_mlp)
test_r2_mlp, test_rmse_mlp, test_mae_mlp = evaluate_model(y_test, y_test_pred_mlp)

print(f'Training R2 (RF, MI): {train_r2_rf_mi:.4f}, RMSE: {train_rmse_rf_mi:.4f}, MAE: {train_mae_rf_mi:.4f}')
print(f'Testing R2 (RF, MI): {test_r2_rf_mi:.4f}, RMSE: {test_rmse_rf_mi:.4f}, MAE: {test_mae_rf_mi:.4f}')

print(f'Training R2 (RF, RFE): {train_r2_rf_rfe:.4f}, RMSE: {train_rmse_rf_rfe:.4f}, MAE: {train_mae_rf_rfe:.4f}')
print(f'Testing R2 (RF, RFE): {test_r2_rf_rfe:.4f}, RMSE: {test_rmse_rf_rfe:.4f}, MAE: {test_mae_rf_rfe:.4f}')

print(f'Training R2 (Linear): {train_r2_linear:.4f}, RMSE: {train_rmse_linear:.4f}, MAE: {train_mae_linear:.4f}')
print(f'Testing R2 (Linear): {test_r2_linear:.4f}, RMSE: {test_rmse_linear:.4f}, MAE: {test_mae_linear:.4f}')

print(f'Training R2 (PCA): {train_r2_pca:.4f}, RMSE: {train_rmse_pca:.4f}, MAE: {train_mae_pca:.4f}')
print(f'Testing R2 (PCA): {test_r2_pca:.4f}, RMSE: {test_rmse_pca:.4f}, MAE: {test_mae_pca:.4f}')

print(f'Training R2 (PLS): {train_r2_pls:.4f}, RMSE: {train_rmse_pls:.4f}, MAE: {train_mae_pls:.4f}')
print(f'Testing R2 (PLS): {test_r2_pls:.4f}, RMSE: {test_rmse_pls:.4f}, MAE: {test_mae_pls:.4f}')

print(f'Training R2 (SVR): {train_r2_svr:.4f}, RMSE: {train_rmse_svr:.4f}, MAE: {train_mae_svr:.4f}')
print(f'Testing R2 (SVR): {test_r2_svr:.4f}, RMSE: {test_rmse_svr:.4f}, MAE: {test_mae_svr:.4f}')

print(f'Training R2 (GBR): {train_r2_gbr:.4f}, RMSE: {train_rmse_gbr:.4f}, MAE: {train_mae_gbr:.4f}')
print(f'Testing R2 (GBR): {test_r2_gbr:.4f}, RMSE: {test_rmse_gbr:.4f}, MAE: {test_mae_gbr:.4f}')

print(f'Training R2 (MLP): {train_r2_mlp:.4f}, RMSE: {train_rmse_mlp:.4f}, MAE: {train_mae_mlp:.4f}')
print(f'Testing R2 (MLP): {test_r2_mlp:.4f}, RMSE: {test_rmse_mlp:.4f}, MAE: {test_mae_mlp:.4f}')

# 交叉验证
cv_scores_ridge_mi = cross_val_score(best_ridge_mi, X_scaled_mi, y_scaled, cv=5, scoring='r2')
cv_scores_ridge_rfe = cross_val_score(best_ridge_rfe, X_scaled_rfe, y_scaled, cv=5, scoring='r2')

cv_scores_rf_mi = cross_val_score(best_rf_mi, X_scaled_mi, y_scaled, cv=5, scoring='r2')
cv_scores_rf_rfe = cross_val_score(best_rf_rfe, X_scaled_rfe, y_scaled, cv=5, scoring='r2')

cv_scores_linear = cross_val_score(linear_reg, X_scaled_mi, y_scaled, cv=5, scoring='r2')
cv_scores_pca = cross_val_score(linear_reg_pca, pca.transform(X_scaled_mi), y_scaled, cv=5, scoring='r2')
cv_scores_pls = cross_val_score(pls, X_scaled_mi, y_scaled, cv=5, scoring='r2')
cv_scores_svr = cross_val_score(best_svr, X_scaled_mi, y_scaled, cv=5, scoring='r2')
cv_scores_gbr = cross_val_score(best_gbr, X_scaled_mi, y_scaled, cv=5, scoring='r2')
cv_scores_mlp = cross_val_score(best_mlp, X_scaled_mi, y_scaled, cv=5, scoring='r2')

print(f'Cross-Validation R2 Scores (Ridge, MI): {cv_scores_ridge_mi}')
print(f'Mean CV R2 Score (Ridge, MI): {np.mean(cv_scores_ridge_mi):.4f}')

print(f'Cross-Validation R2 Scores (Ridge, RFE): {cv_scores_ridge_rfe}')
print(f'Mean CV R2 Score (Ridge, RFE): {np.mean(cv_scores_ridge_rfe):.4f}')

print(f'Cross-Validation R2 Scores (RF, MI): {cv_scores_rf_mi}')
print(f'Mean CV R2 Score (RF, MI): {np.mean(cv_scores_rf_mi):.4f}')

print(f'Cross-Validation R2 Scores (RF, RFE): {cv_scores_rf_rfe}')
print(f'Mean CV R2 Score (RF, RFE): {np.mean(cv_scores_rf_rfe):.4f}')

print(f'Cross-Validation R2 Scores (Linear): {cv_scores_linear}')
print(f'Mean CV R2 Score (Linear): {np.mean(cv_scores_linear):.4f}')

print(f'Cross-Validation R2 Scores (PCA): {cv_scores_pca}')
print(f'Mean CV R2 Score (PCA): {np.mean(cv_scores_pca):.4f}')

print(f'Cross-Validation R2 Scores (PLS): {cv_scores_pls}')
print(f'Mean CV R2 Score (PLS): {np.mean(cv_scores_pls):.4f}')

print(f'Cross-Validation R2 Scores (SVR): {cv_scores_svr}')
print(f'Mean CV R2 Score (SVR): {np.mean(cv_scores_svr):.4f}')

print(f'Cross-Validation R2 Scores (GBR): {cv_scores_gbr}')
print(f'Mean CV R2 Score (GBR): {np.mean(cv_scores_gbr):.4f}')

print(f'Cross-Validation R2 Scores (MLP): {cv_scores_mlp}')
print(f'Mean CV R2 Score (MLP): {np.mean(cv_scores_mlp):.4f}')

# 逆标准化处理
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_train_pred_ridge_mi_inv = scaler_y.inverse_transform(y_train_pred_ridge_mi.reshape(-1, 1)).flatten()
y_test_pred_ridge_mi_inv = scaler_y.inverse_transform(y_test_pred_ridge_mi.reshape(-1, 1)).flatten()
y_train_pred_ridge_rfe_inv = scaler_y.inverse_transform(y_train_pred_ridge_rfe.reshape(-1, 1)).flatten()
y_test_pred_ridge_rfe_inv = scaler_y.inverse_transform(y_test_pred_ridge_rfe.reshape(-1, 1)).flatten()
y_train_pred_rf_mi_inv = scaler_y.inverse_transform(y_train_pred_rf_mi.reshape(-1, 1)).flatten()
y_test_pred_rf_mi_inv = scaler_y.inverse_transform(y_test_pred_rf_mi.reshape(-1, 1)).flatten()
y_train_pred_rf_rfe_inv = scaler_y.inverse_transform(y_train_pred_rf_rfe.reshape(-1, 1)).flatten()
y_test_pred_rf_rfe_inv = scaler_y.inverse_transform(y_test_pred_rf_rfe.reshape(-1, 1)).flatten()
y_train_pred_linear_inv = scaler_y.inverse_transform(y_train_pred_linear.reshape(-1, 1)).flatten()
y_test_pred_linear_inv = scaler_y.inverse_transform(y_test_pred_linear.reshape(-1, 1)).flatten()
y_train_pred_pca_inv = scaler_y.inverse_transform(y_train_pred_pca.reshape(-1, 1)).flatten()
y_test_pred_pca_inv = scaler_y.inverse_transform(y_test_pred_pca.reshape(-1, 1)).flatten()
y_train_pred_pls_inv = scaler_y.inverse_transform(y_train_pred_pls.reshape(-1, 1)).flatten()
y_test_pred_pls_inv = scaler_y.inverse_transform(y_test_pred_pls.reshape(-1, 1)).flatten()
y_train_pred_svr_inv = scaler_y.inverse_transform(y_train_pred_svr.reshape(-1, 1)).flatten()
y_test_pred_svr_inv = scaler_y.inverse_transform(y_test_pred_svr.reshape(-1, 1)).flatten()
y_train_pred_gbr_inv = scaler_y.inverse_transform(y_train_pred_gbr.reshape(-1, 1)).flatten()
y_test_pred_gbr_inv = scaler_y.inverse_transform(y_test_pred_gbr.reshape(-1, 1)).flatten()
y_train_pred_mlp_inv = scaler_y.inverse_transform(y_train_pred_mlp.reshape(-1, 1)).flatten()
y_test_pred_mlp_inv = scaler_y.inverse_transform(y_test_pred_mlp.reshape(-1, 1)).flatten()

# 保存训练集实际值和预测值到Excel文件
train_results = {
    'Actual Train': y_train_inv,
    'Predicted Train (Ridge, MI)': y_train_pred_ridge_mi_inv,
    'Predicted Train (Ridge, RFE)': y_train_pred_ridge_rfe_inv,
    'Predicted Train (RF, MI)': y_train_pred_rf_mi_inv,
    'Predicted Train (RF, RFE)': y_train_pred_rf_rfe_inv,
    'Predicted Train (Linear)': y_train_pred_linear_inv,
    'Predicted Train (PCA)': y_train_pred_pca_inv,
    'Predicted Train (PLS)': y_train_pred_pls_inv,
    'Predicted Train (SVR)': y_train_pred_svr_inv,
    'Predicted Train (GBR)': y_train_pred_gbr_inv,
    'Predicted Train (MLP)': y_train_pred_mlp_inv
}

train_results_df = pd.DataFrame(train_results)
train_results_df.to_excel('训练集实际值和预测值.xlsx', index=False)

# 保存测试集实际值和预测值到Excel文件
test_results = {
    'Actual Test': y_test_inv,
    'Predicted Test (Ridge, MI)': y_test_pred_ridge_mi_inv,
    'Predicted Test (Ridge, RFE)': y_test_pred_ridge_rfe_inv,
    'Predicted Test (RF, MI)': y_test_pred_rf_mi_inv,
    'Predicted Test (RF, RFE)': y_test_pred_rf_rfe_inv,
    'Predicted Test (Linear)': y_test_pred_linear_inv,
    'Predicted Test (PCA)': y_test_pred_pca_inv,
    'Predicted Test (PLS)': y_test_pred_pls_inv,
    'Predicted Test (SVR)': y_test_pred_svr_inv,
    'Predicted Test (GBR)': y_test_pred_gbr_inv,
    'Predicted Test (MLP)': y_test_pred_mlp_inv
}

test_results_df = pd.DataFrame(test_results)
test_results_df.to_excel('测试集实际值和预测值.xlsx', index=False)

# 可视化结果
fig, axes = plt.subplots(6, 2, figsize=(15, 30))

# 绘制训练集结果
axes[0, 0].plot(y_train_inv, label='Actual Train')
axes[0, 0].plot(y_train_pred_ridge_mi_inv, label='Predicted Train (Ridge, MI)')
axes[0, 0].legend()
axes[0, 0].set_title('Training Set: Actual vs Predicted (Ridge, MI)')

axes[1, 0].plot(y_train_inv, label='Actual Train')
axes[1, 0].plot(y_train_pred_ridge_rfe_inv, label='Predicted Train (Ridge, RFE)')
axes[1, 0].legend()
axes[1, 0].set_title('Training Set: Actual vs Predicted (Ridge, RFE)')

axes[2, 0].plot(y_train_inv, label='Actual Train')
axes[2, 0].plot(y_train_pred_rf_mi_inv, label='Predicted Train (RF, MI)')
axes[2, 0].legend()
axes[2, 0].set_title('Training Set: Actual vs Predicted (RF, MI)')

axes[3, 0].plot(y_train_inv, label='Actual Train')
axes[3, 0].plot(y_train_pred_rf_rfe_inv, label='Predicted Train (RF, RFE)')
axes[3, 0].legend()
axes[3, 0].set_title('Training Set: Actual vs Predicted (RF, RFE)')

axes[4, 0].plot(y_train_inv, label='Actual Train')
axes[4, 0].plot(y_train_pred_linear_inv, label='Predicted Train (Linear)')
axes[4, 0].legend()
axes[4, 0].set_title('Training Set: Actual vs Predicted (Linear)')

axes[5, 0].plot(y_train_inv, label='Actual Train')
axes[5, 0].plot(y_train_pred_pca_inv, label='Predicted Train (PCA)')
axes[5, 0].legend()
axes[5, 0].set_title('Training Set: Actual vs Predicted (PCA)')

# 绘制测试集结果
axes[0, 1].plot(y_test_inv, label='Actual Test')
axes[0, 1].plot(y_test_pred_ridge_mi_inv, label='Predicted Test (Ridge, MI)')
axes[0, 1].legend()
axes[0, 1].set_title('Test Set: Actual vs Predicted (Ridge, MI)')

axes[1, 1].plot(y_test_inv, label='Actual Test')
axes[1, 1].plot(y_test_pred_ridge_rfe_inv, label='Predicted Test (Ridge, RFE)')
axes[1, 1].legend()
axes[1, 1].set_title('Test Set: Actual vs Predicted (Ridge, RFE)')

axes[2, 1].plot(y_test_inv, label='Actual Test')
axes[2, 1].plot(y_test_pred_rf_mi_inv, label='Predicted Test (RF, MI)')
axes[2, 1].legend()
axes[2, 1].set_title('Test Set: Actual vs Predicted (RF, MI)')

axes[3, 1].plot(y_test_inv, label='Actual Test')
axes[3, 1].plot(y_test_pred_rf_rfe_inv, label='Predicted Test (RF, RFE)')
axes[3, 1].legend()
axes[3, 1].set_title('Test Set: Actual vs Predicted (RF, RFE)')

axes[4, 1].plot(y_test_inv, label='Actual Test')
axes[4, 1].plot(y_test_pred_linear_inv, label='Predicted Test (Linear)')
axes[4, 1].legend()
axes[4, 1].set_title('Test Set: Actual vs Predicted (Linear)')

axes[5, 1].plot(y_test_pred_pca_inv, label='Predicted Test (PCA)')
axes[5, 1].legend()
axes[5, 1].set_title('Test Set: Actual vs Predicted (PCA)')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(6, 2, figsize=(15, 30))

# 绘制训练集结果
axes[0, 0].plot(y_train_inv, label='Actual Train')
axes[0, 0].plot(y_train_pred_pls_inv, label='Predicted Train (PLS)')
axes[0, 0].legend()
axes[0, 0].set_title('Training Set: Actual vs Predicted (PLS)')

axes[1, 0].plot(y_train_inv, label='Actual Train')
axes[1, 0].plot(y_train_pred_svr_inv, label='Predicted Train (SVR)')
axes[1, 0].legend()
axes[1, 0].set_title('Training Set: Actual vs Predicted (SVR)')

axes[2, 0].plot(y_train_inv, label='Actual Train')
axes[2, 0].plot(y_train_pred_gbr_inv, label='Predicted Train (GBR)')
axes[2, 0].legend()
axes[2, 0].set_title('Training Set: Actual vs Predicted (GBR)')

axes[3, 0].plot(y_train_inv, label='Actual Train')
axes[3, 0].plot(y_train_pred_mlp_inv, label='Predicted Train (MLP)')
axes[3, 0].legend()
axes[3, 0].set_title('Training Set: Actual vs Predicted (MLP)')

# 绘制测试集结果
axes[0, 1].plot(y_test_inv, label='Actual Test')
axes[0, 1].plot(y_test_pred_pls_inv, label='Predicted Test (PLS)')
axes[0, 1].legend()
axes[0, 1].set_title('Test Set: Actual vs Predicted (PLS)')

axes[1, 1].plot(y_test_inv, label='Actual Test')
axes[1, 1].plot(y_test_pred_svr_inv, label='Predicted Test (SVR)')
axes[1, 1].legend()
axes[1, 1].set_title('Test Set: Actual vs Predicted (SVR)')

axes[2, 1].plot(y_test_inv, label='Actual Test')
axes[2, 1].plot(y_test_pred_gbr_inv, label='Predicted Test (GBR)')
axes[2, 1].legend()
axes[2, 1].set_title('Test Set: Actual vs Predicted (GBR)')

axes[3, 1].plot(y_test_inv, label='Actual Test')
axes[3, 1].plot(y_test_pred_mlp_inv, label='Predicted Test (MLP)')
axes[3, 1].legend()
axes[3, 1].set_title('Test Set: Actual vs Predicted (MLP)')

plt.tight_layout()
plt.show()
