from __future__ import division
import numpy as np
#hmmlearn中主要有两种模型，分布为：GaussianHMM和MultinomialHMM；
#如果观测值是连续的，那么建议使用GaussianHMM，否则使用MultinomialHMM；
from hmmlearn import hmm
states = ["Rainy", "Sunny"]##隐藏状态
n_states = len(states)##长度

observations = ["walk", "shop", "clean"]##可观察的状态
n_observations = len(observations)##可观察序列的长度

start_probability = np.array([0.6, 0.4])##开始转移概率
##转移矩阵
transition_probability = np.array([
  [0.7, 0.3],
  [0.4, 0.6]
])
##混淆矩阵
emission_probability = np.array([
  [0.1, 0.4, 0.5],
  [0.6, 0.3, 0.1]
])

#构建了一个MultinomialHMM模型，这模型包括开始的转移概率，隐含间的转移矩阵A（transmat），隐含层到可视层的混淆矩阵emissionprob，下面是参数初始化
model = hmm.MultinomialHMM(n_components=n_states)
model.set_startprob(start_probability)
model.set_transmat(transition_probability)
model.set_emissionprob(emission_probability)

# predict a sequence of hidden states based on visible states
bob_says = [2, 2, 1, 1, 2, 2]##预测时的可见序列
logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
print (logprob)##该参数反映模型拟合的好坏
##最后输出结果
print ("Bob says:", ", ".join(map(lambda x: observations[x], bob_says)))
print ("Alice hears:", ", ".join(map(lambda x: states[x], alice_hears)))
