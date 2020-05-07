import numpy as np
from matplotlib import pyplot as plt
# 从sklearn中导入svm中的SVC
from sklearn.svm import SVC
# 导入sklearn的数据案例中的鸢尾花数据集
from sklearn.datasets import load_iris
# 读取数据，不需要写路径
data = load_iris()
#关于data和target是由数据集写的
X = data['data']
y = data['target']
print(X)
print(y)
# 数据提取（该数据了的种类由0、1、2三个标签，本次只选择0、1这个数据，我也不知道为啥- -，网上大部分都去掉2）
X = X[y!= 2,0:2]
y = y[y!= 2]
#Z-score标准化方法：（x-平均数）/标准差
#X=preprocessing.scale(X)
#print(np.mean(X,axis=0))
#手动实现Z-score标准化
X -= np.mean(X,axis=0)
X /= np.std(X,axis=0,ddof=1)
#print(X)
m = len(X)
# 数据切割8：2
d = int(0.8 * m)
X_train,X_test = np.split(X,[d])
y_train,y_test = np.split(y,[d])
# 创建SVM模型
model_svm = SVC(C=1,kernel='rbf')
# 调用fit函数训练模型
model_svm.fit(X_train,y_train)
# 查看准确率
ss = model_svm.score(X_test,y_test)
print('测试集的准确率是：',ss)
# 调用训练好的模型获得预测的值
X_train_h = model_svm.predict(X_train)
X_test_h = model_svm.predict(X_test)

# 开始画图部分
# 确定画图的范围
x1_min,x1_max,x2_min,x2_max = np.min(X[:,0]),np.max(X[:,0]),np.min(X[:,1]),np.max(X[:,1])
# 将画布切割成200*200
x1,x2 = np.mgrid[x1_min:x1_max:200j,x2_min:x2_max:200j]
# 计算点到超平面的距离
# 首先对数据进行拼接
x1x2 = np.c_[x1.ravel(),x2.ravel()]
z = model_svm.decision_function(x1x2)
z = z.reshape(x1.shape)
# 画出所有的样本点
plt.scatter(X[:,0],X[:,1],c=y,zorder=10)
# 画出测试集的样本点
plt.scatter(X_train[:,0],X_train[:,1],s=100,facecolor='none',zorder=10,edgecolors='k')
# 画等值面
# plt.cm中cm全称表示colormap，
# paired表示两个两个相近色彩输出，比如浅蓝、深蓝；浅红、深红；浅绿，深绿这种。
plt.contourf(x1,x2,z>=0,cmap=plt.cm.Paired)
# 画等值线
plt.contour(x1,x2,z,levels=[-1,0,1])
plt.show()

