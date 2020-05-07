import json
import operator
#字典
dict={'a':1,'b':2,'c':3}
print(dict)
print(type(dict))
#json.dumps()用于将字典形式的数据转化为字符串，json.loads()用于将字符串形式的数据转化为字典
#如果不转化，提取a，对于字符串只能用正则表达式提取
json1=json.dumps(dict)
print(json1)
print(type(json1))
str1='{"a":1,"b":2,"c":3}'
dict2=json.loads(str1)
print(dict2)
print(type(dict2))

print(dict2["a"])
for i in dict2.keys():
    print(i)
for i in dict2.values():
    print(i)
#相等判定
print(operator.eq(dict,dict2))