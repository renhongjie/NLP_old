from NLP1.cnn_lstm.tools.segment import Segment
from NLP1.cnn_lstm.train_process import TrainParam
import train_model
from NLP1.cnn_lstm.tools import common

seg = Segment(
    stopword_files=["stopwords/1.txt", "stopwords/2.txt", "stopwords/3.txt", "stopwords/4.txt"],
    jieba_tmp_dir="my_tmp"
)


def tokenizer(text: str):
    text = common.normalize(text)
    return seg.cut(text)


class Word2vecIterator(object):
    def __iter__(self):
        with open("Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt", "r", encoding="utf-8") as rf:
            row = rf.readline()
            row_size = int(row.strip().split(" ")[0])
            for i, row in enumerate(rf):
                if i % 10000 == 0:
                    print("%.f%% === %d/%d\r" % ((i + 1) * 100.0 / row_size, i + 1, row_size), end="")
                ls = row.strip().split(" ")
                word = ls[0]
                vec = ls[1:]
                yield word, vec
            print()


if __name__ == "__main__":
    config = TrainParam()
    config.output_dir = "politics_output.word2vec4"  #输出目录
    config.tokenizer = tokenizer                        #分词函数
    config.cate_list = ["normal", "politics"]           #分类类别
    config.data_dir = "dataset"  #训练数据目录
    config.kernel_size_list = [3, 4, 5]
    config.epoches = 5                                  #训练轮数
    config.cuda = False                                  #是否使用gpu训练
    config.model_name = "politics"                      #模型名称
    config.data_vocab_dir = "dataset/politics"  #提取词汇表目录
    config.max_vocab_size = 20000                       #词汇表最大词数
    config.min_word_freq = 10                           #词频最小值，大于该值才进入词汇表
    config.is_static_word2vec = False
    config.dropout = 0.5
    config.word2vec_iterator = Word2vecIterator()       #训练好的Word2vec模型
    config.seg_sentence = False
    config.max_sentence_len = 300                       #句子最大词数
    config.embed_dim = 200
    config.kernel_num = 100
    config.num_hiddens = 128
    config.num_layers = 4
    config.bidirectional = True
    config.train_batch_size = 128
    train_model.train(config, is_valid_only=False)
