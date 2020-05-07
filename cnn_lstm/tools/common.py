# -*- coding: UTF-8 -*-

from NLP1.cnn_lstm.tools.langconv import *
import codecs
import re
def normalize(tag):
    regEx_html = "<[^>]+>" # 定义HTML标签的正则表达式
    tag = re.sub(regEx_html,'',tag)
    tag = re.sub("[0-9]*/[0-9]*/[0-9]*", '', tag)
    tag = re.sub("《大纪元.*》", '', tag)
    tag = re.sub("\.", '', tag)
    tag = re.sub("\t\n",'',tag)
    tag = re.sub("-", '', tag)
    tag = re.sub("((\r\n)|\n)[\\s\t ]*(\\1)+", '', tag)
    tag = re.sub("^((\r\n)|\n)+", '', tag)
    tag = re.sub("    +| +|　+", '', tag)
    tag = re.sub("<br>", '', tag)
    tag = re.sub("转载|首发|喜讯|推荐|编辑|人气|禁闻|大纪元|新唐人|文化大观|翻墙必看|健康医疗|一周大事解读|热点透视|quot;|lt;|amp;|brgt;", '', tag)
    tag = re.sub("(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", '', tag)
    tag = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】《》“”〝〞．！：，:；★‘’﹕〈〉。？、⋯~@#￥%…┅&*（）「『」』－～＊︰║‥①②③④⑤⑥⑦⑧⑨\[\]�]", "", tag)
    tag = Converter('zh-hans').convert(tag)
    return tag

def save_txt(data,save_path,tag):
    with codecs.open(save_path, 'wb', 'utf-8') as out:
        for line in data:
            out.write(tag + ',' + normalize(str(line)) + '\n')

