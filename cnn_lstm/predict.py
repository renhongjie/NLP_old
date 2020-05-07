from NLP1.cnn_lstm.predict_model import ModelPredict
from NLP1.cnn_lstm.tools import common
from NLP1.cnn_lstm.tools.segment import Segment
import os
import time
import codecs

base_dir = os.path.dirname(os.path.realpath(__file__))

stopword_files = []
for root, _, fnames in os.walk(os.path.join(base_dir, "stopwords")):
    for fname in fnames:
        fpath = os.path.join(root, fname)
        stopword_files.append(fpath)

seg = Segment(stopword_files=stopword_files)


def tokenizer(text: str) -> list:
    text = common.normalize(text)
    return seg.cut(text)


def predict(text: str) -> int:
    label, proba = pred_model.predict(text)
    return label


def predict_txt(test_path: str):
    count = 0
    ac = 0
    p_sum = 0
    p_ac = 0
    n_sum = 0
    n_ac = 0
    err_results = []
    start = time.time()
    with open(test_path, 'r', encoding='utf-8') as f:
        line = f.readline().rstrip()
        while line:
            count += 1
            if count % 1000 == 0:
                print(count)
                # break
            tag = line[0:]
            line = line[2:]
            if line == None or line == '':
                line = f.readline().rstrip()
                continue
            label = predict(common.normalize(line))
            # label = label[0][0]
            if str(tag) == '0':
                n_sum += 1
            elif str(tag) == '1':
                p_sum += 1
            if str(label) == '0' and str(label) == tag:
                n_ac += 1
            elif str(label) == '1' and str(label) == tag:
                p_ac += 1
            if str(label) == tag:
                ac += 1
            else:
                err_results.append("预测结果：" + str(label) + "，标签：" + tag + "," + line)
            line = f.readline().rstrip()
    end = time.time()
    times = end - start
    # output
    with codecs.open('test_summary.txt', 'wb', 'utf-8') as out:
        out.write(
            "正确率:" + str((ac / count) * 100) + '%,平均时间:' + str(times / count) + '秒,,涉证拒绝率:{0},正常文本通过率:{1}\n'.format(
                p_ac / p_sum, n_ac / n_sum))
        print(p_ac, p_sum)
        print(n_ac, n_sum)
        print(ac, count)
        out.write("================以下数据判断错误=====================\n")
        for line in err_results:
            out.write(line + '\n')


if __name__ == '__main__':
    pred_model = ModelPredict(os.path.join(base_dir, "politics_output.word2vec4/politics.pt"),
                                os.path.join(base_dir, "politics_output.word2vec4/vocab"), tokenizer)
    predict_txt(r'D:\doc\知识社群审核语料库\涉黄\sex_data.txt')