import numpy as np
import torch
import os


class TrainConfig(object):
    def __init__(self):
        # --------数据处理相关---------------
        self.seg_sentence = False  # 大文档是否按照max_sentence_len拆分成多个小文档
        self.word2vec_iterator = None
        self.is_static_word2vec = False
        self.max_sentence_len = 10000
        self.min_word_freq = 5
        self.max_vocab_size = np.inf
        self.test_rate = 0.1
        self.output_dir = ""
        # 分词函数
        self.tokenizer = None
        # 训练样本目录
        self.data_dir = "dataset"
        # 提取词汇表目录
        self.data_vocab_dir = "dataset/politics"
        # 分类类别列表
        self.cate_list = []

        # -------模型配置相关------------------
        # 词向量维度
        self.embed_dim = 100
        # 卷积核数量
        self.kernel_num = 100
        # 卷积核步长列表
        self.kernel_size_list = [3, 4, 5]
        self.dropout = 0.5
        self.num_hiddens = 128
        self.num_layers = 1
        self.bidirectional = True

        # -------训练参数相关------------------
        self.learning_rate = 0.001
        self.epoches = 0
        self.cuda = True
        self.log_interval = 10
        self.test_interval = 50
        self.save_interval = 100
        self.train_batch_size = 64
        self.test_batch_size = 100
        self.model_name = "cnnlstmmodel"
        self.continue_train = True


def train(config: TrainConfig, is_valid_only=False):
    from NLP1.cnn_lstm.dataset import DatasetParam, Dataset
    from NLP1.cnn_lstm.model import ModelParam
    from NLP1.cnn_lstm.train_process import TrainParam, ModelTrain
    import logging
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    fmt = "%(asctime)s:%(filename)s:%(funcName)s:%(lineno)s: %(levelname)s - %(message)s"
    logging.basicConfig(filename=os.path.join(config.output_dir, "train.log"), format=fmt, level=logging.DEBUG)

    # 数据预处理
    dataset_args = DatasetParam()
    dataset_args.output_dir = config.output_dir
    dataset_args.embed_dim = config.embed_dim
    dataset_args.max_sentence_len = config.max_sentence_len
    dataset_args.min_word_freq = config.min_word_freq
    dataset_args.max_vocab_size = config.max_vocab_size
    dataset_args.test_rate = config.test_rate
    dataset_args.tokenizer = config.tokenizer
    dataset_args.data_dir = config.data_dir
    dataset_args.cate_list = config.cate_list
    dataset_args.word2vec_iterator = config.word2vec_iterator
    dataset_args.data_vocab_dir = config.data_vocab_dir
    dataset_args.num_hiddens = config.num_hiddens
    dataset_args.num_layers = config.num_layers
    dataset = Dataset(dataset_args)
    data_iter, vocab_dict, weights = dataset.build(config.seg_sentence)

    # 初始化模型参数
    model_args = ModelParam()
    model_args.is_static_word2vec = config.is_static_word2vec
    model_args.weights = weights
    model_args.class_num = len(config.cate_list)
    model_args.vocab_size = len(vocab_dict)
    model_args.embed_dim = config.embed_dim
    model_args.kernel_num = config.kernel_num
    model_args.kernel_size_list = config.kernel_size_list
    model_args.dropout = config.dropout

    # 初始化训练参数
    train_args = TrainParam()
    train_args.learning_rate = config.learning_rate
    train_args.epoches = config.epoches
    train_args.cuda = config.cuda
    train_args.log_interval = config.log_interval
    train_args.test_interval = config.test_interval
    train_args.save_interval = config.save_interval
    train_args.train_batch_size = config.train_batch_size
    train_args.test_batch_size = config.test_batch_size
    train_args.model_save_dir = config.output_dir
    train_args.model_name = config.model_name
    train_args.continue_train = config.continue_train

    model_train = ModelTrain(train_args, model_args)

    # 训练
    if not is_valid_only:
        model_train.train(data_iter)
    model_train.valid(data_iter)
