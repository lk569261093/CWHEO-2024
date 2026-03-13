import pandas as pd

# 读取Excel文件
file_path = '化验数据及对应光谱数据.xlsx'
df = pd.read_excel(file_path)

# 获取光谱数据的列名
spectral_columns = df.columns[3:]  # 从第四列开始是光谱数据

# 将光谱数据除以10000.0
df[spectral_columns] = df[spectral_columns] / 10000.0

# 保存处理后的数据到新的Excel文件
output_file_path = '化验数据及对应反射率数据.xlsx'
df.to_excel(output_file_path, index=False)

print(f"处理后的数据已保存到 {output_file_path}")
