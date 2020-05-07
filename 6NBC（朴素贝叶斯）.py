from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
import jieba
import os
#读取文件的内容并进行分词
def load_txt(path):
    #= ""是为了确定数据类型
    text = ""
    textfile = open(path, "r", encoding="utf8").read()
    #运用jieba分词进行切割
    textcut = jieba.cut(textfile)
    for word in textcut:
        text += word + " "
    return text
#读取所有文件内容
#因为一个文件夹的标签是一样的，所有把标签当成一个参数，对该文件夹下的所有txt文件都赋该classtag
def load_alltxt(path, classtag):
    allfiles = os.listdir(path)
    allclasstags = []
    processed_textset = []

    for thisfile in allfiles:
        path_name = path + "/" + thisfile
        processed_textset.append(load_txt(path_name))
        allclasstags.append(classtag)
    return processed_textset, allclasstags

textdata1, class1 = load_alltxt("/Users/ren/PycharmProjects/人工智能/NLP1/datas/hotel", "hotel")
textdata2, class2 = load_alltxt("/Users/ren/PycharmProjects/人工智能/NLP1/datas/travel", "travel")
train_data = textdata1 + textdata2
classtags_list = class1 + class2
#CountVectorizer是属于常见的特征数值计算类，是一个文本特征提取方法。对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率。
#CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数。
count_vector = CountVectorizer()
#CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵
#词典的建立
vector_matrix = count_vector.fit_transform(train_data)
#特征的提取
#TfidfTransformer用于统计vectorizer中每个词语的TF-IDF值。
train_tfidf = TfidfTransformer(use_idf=True).fit_transform(vector_matrix)
#分类器朴素贝叶斯，除了MultinomialNB之外，还有GaussianNB就是先验为高斯分布的朴素贝叶斯，BernoulliNB就是先验为伯努利分布的朴素贝叶斯。
#clf =MultinomialNB().fit(train_tfidf, classtags_list)
#伯努利贝叶斯分类器
clf = BernoulliNB().fit(train_tfidf, classtags_list)
#测试文件的路径
path = "/Users/ren/PycharmProjects/人工智能/NLP1/datas/NBC-text"
allfiles = os.listdir(path)

print(allfiles)
for thisfile in allfiles:
    #一直报错，输出后发现['2-travel.txt', '.DS_Store', '3-hotel.txt', '4-travel.txt', '1-travel.txt']
    #.DS_Store为Mac系统的一个文件，自动生成的，很烦人，所有加了个判断，如果文件名不含txt就跳过
    if 'txt' not in thisfile:
        continue
    path_name = path + "/" + thisfile
    #将预测文件变成词频矩阵
    new_count_vector = count_vector.transform([load_txt(path_name)])
    # 将词频矩阵变成TF-IDF值
    new_tfidf = TfidfTransformer(use_idf=True).fit_transform(new_count_vector)

    predict_result = clf.predict(new_tfidf)
    print("文件名字：",thisfile,"预测tag：",predict_result)
    print("-------------------")