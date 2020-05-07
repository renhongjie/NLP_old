import os
import scrapy
# 读取文件file已经快被淘汰了，现在都用open，新手写法：
filename="/Users/ren/PycharmProjects/人工智能/qingmo/2.py"
file=open(filename,"r")
S=file.read()
print(S)
#企业的写法,with不需要close
with open(filename,"r",encoding='utf8') as f:
    s=f.read()
    print(s)
#企业写文件
logfile=open("/Users/ren/PycharmProjects/人工智能/qingmo/renhongjie.txt",'w',encoding='utf8')
for i in range(100):
    print(i,file=logfile)
logfile.close()#不关闭的话，直接读取该文件则文件内没有任何信息
if __name__ == '__main__':
    path="/Users/ren/PycharmProjects/人工智能/qingmo"
    allfiles=os.listdir(path)#读取该路径下所有有文件
    print(path)
    for i in allfiles:
        print(i)
        if i=='renhongjie.txt':
            filename2=path+'/'+i
            with open(filename2,"r",encoding='utf8') as f2:
                s2=f2.read()#常用的read readline readlines（存到list）
                print(filename2,s2,"***")
