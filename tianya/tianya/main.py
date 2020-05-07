#两种启动爬虫的方法,如下
#一个执行cmd的库
from scrapy import cmdline
#1调用cmd
#cmdline.execute("scrapy crawl tianya".split())
from scrapy.cmdline import execute
#2直接给execute传参数
execute(["scrapy","crawl","tianya"])