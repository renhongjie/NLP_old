import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class ModelParam(object):
    def __init__(self):
        # 词向量是否不更新
        self.is_static_word2vec = False
        # 初始词向量权重
        self.weights = None
        # 分类类型
        self.class_num = 2
        # 词汇表大小
        self.vocab_size = 0
        # 词向量维度
        self.embed_dim = 0
        # 卷积核数量
        self.kernel_num = 0
        # 卷积核步长列表
        self.kernel_size_list = []
        # lstm hidden unit
        self.num_hiddens = 128
        # 有几层 lstm layers
        self.num_layers = 2
        # 是否双向
        self.bidirectional = True
        # dropout
        self.dropout = 0.2

    @classmethod
    def from_dict(cls, param: dict):
        obj = ModelParam()
        for k, v in param.items():
            obj.__setattr__(k, v)
        return obj


class Model(nn.Module):
    def __init__(self, args: ModelParam):
        super(Model, self).__init__()
        self.args = args

        weights = None
        if self.args.weights is not None:
            if isinstance(self.args.weights, np.ndarray):
                weights = torch.from_numpy(self.args.weights).float()
            elif isinstance(self.args.weights, list):
                weights = torch.FloatTensor(self.args.weights)

            # 这个参数比较大，而一次性使用，置空可以压缩模型保存大小
            self.args.weights = None

        self.embed = nn.Embedding(self.args.vocab_size, self.args.embed_dim, _weight=weights)

        in_channel = 1
        # 输出通道数是卷积核的数量
        out_channel = self.args.kernel_num
        embed_dim = self.args.embed_dim
        KS = self.args.kernel_size_list
        self.convslist = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embed_dim)) for K in KS])
        self.dropout = nn.Dropout(self.args.dropout, inplace=True)

        # LSTM / GRU
        self.encoder = nn.LSTM(input_size=out_channel, hidden_size=self.args.num_hiddens,
                               num_layers=self.args.num_layers, bidirectional=self.args.bidirectional)

        #是否双向
        if self.args.bidirectional:
            self.decoder = nn.Linear(self.args.num_hiddens * self.args.num_layers * 2, self.args.class_num)
        else:
            self.decoder = nn.Linear(self.args.num_hiddens * self.args.num_layers, self.args.class_num)

    def get_predict_args(self) -> dict:
        args = self.args
        args.weights = None
        args.dropout = 1.0
        return args.__dict__

    def forward(self, input_x):
        out_channel = self.args.kernel_num
        x = self.embed(input_x)
        N, W, D = tuple(x.size())
        x = x.unsqueeze(1)
        if self.args.is_static_word2vec:
            x = Variable(x, reqires_grad=False)
        # encoder
        x = [F.relu(conv(x), inplace=True).squeeze(3) for conv in self.convslist]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        # decoder
        x = x.view(N, -1, out_channel)
        #状态和隐藏层
        states, hidden = self.encoder(x.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs
