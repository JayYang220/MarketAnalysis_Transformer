from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 创建MinMaxScaler对象
scaler = MinMaxScaler()

# 示例数据
data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

# 縮放
sc = MinMaxScaler(feature_range=(-1, 1))
# 将数据传递给MinMaxScaler对象并进行缩放
scaled_data = sc.fit_transform(data)

print("原始数据：\n", data)
print("缩放后的数据：\n", scaled_data)
print(type(scaled_data))

a = [[1], [2], [3], [3]]
a = np.array(a)

print(a.shape[1])