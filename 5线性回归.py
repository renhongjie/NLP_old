import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
#读数据
dataframe=pd.read_fwf('/Users/ren/PycharmProjects/人工智能/NLP1/datas/brain_body.txt')
x_values=dataframe[['Brain']]
y_values=dataframe[['Body']]
#训练模型
body_reg=linear_model.LinearRegression()
body_reg.fit(x_values,y_values)
#可视化
plt.scatter(x_values,y_values)
plt.plot(x_values,body_reg.predict(x_values))
plt.show()