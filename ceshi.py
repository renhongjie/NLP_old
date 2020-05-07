# 导入科学计算包，用于矩阵计算或数据拼接
import numpy as np
# 导入画图工具
from matplotlib import pyplot as plt
# 从sklearn中导入svm中的SVC
from sklearn.svm import SVC
# 导入sklearn的数据案例中的鸢尾花数据集
from sklearn.datasets import load_iris

from sklearn import preprocessing
# 读取数据
data = load_iris()


# 定义数据处理函数

X = data['data']
y = data['target']
print(X)
print(y)
# 数据提取
X = X[y!= 2,0:2]
y = y[y!= 2]
#Z-score标准化方法：（x-平均数）/标准差
#X=preprocessing.scale(X)
print(np.mean(X,axis=0))
X -= np.mean(X,axis=0)
X /= np.std(X,axis=0,ddof=1)

print(X)
m = len(X)
# 数据切割
d = int(0.7 * m)
X_train,X_test = np.split(X,[d])
y_train,y_test = np.split(y,[d])

