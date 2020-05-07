#encoding=utf-8
import codecs
import os

with open(r'D:\doc\知识社群审核语料库\train_data.txt','r',encoding='utf-8') as f:
    index = 1
    line = f.readline().rstrip()
    while line:
        tag, line = line[0], line[2:]
        if not os.path.exists('../dataset/{0}'.format(tag)):
            os.makedirs('../dataset/{0}'.format(tag))
        with codecs.open('../dataset/{0}/{1}.txt'.format(tag, index), 'wb', 'utf-8') as out:
            out.write(line + '\n')
        print(index)
        index += 1
        line = f.readline().rstrip()