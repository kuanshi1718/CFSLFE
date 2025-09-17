import numpy as np
import pandas as pd

# 读取npy文件
data = np.load('C:/Users/Administrator/Desktop/3weights.npy')

# 获取数据维度
n_dims = data.ndim

# 创建多维索引
index = pd.MultiIndex.from_tuples(
    [(i, j, k, l) for i in range(data.shape[0])
     for j in range(data.shape[1])
     for k in range(data.shape[2])
     for l in range(data.shape[3])]
)

# 将numpy数组转换为pandas的Series
series = pd.Series(data.flatten(), index=index)

# 将Series转换为DataFrame
df = series.unstack()

# 将DataFrame分割为多个部分
chunk_size = 10000  # 每个工作表的最大行数
chunks = [df.iloc[i:i + chunk_size, :] for i in range(0, len(df), chunk_size)]

# 保存为多个Excel工作表
with pd.ExcelWriter('C:/Users/Administrator/Desktop/output.xlsx') as writer:
    for i, chunk in enumerate(chunks):
        chunk.to_excel(writer, sheet_name=f'Sheet_{i + 1}', index=False)