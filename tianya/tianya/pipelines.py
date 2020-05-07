# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymysql
#数据库操作
class TianyaPipeline(object):
    def __init__(self):
        # 连接MySQL数据库
        print("正在连接")
        self.connect = pymysql.connect(host='localhost', user='root', password='rhjfxy', db='data', port=3306)
        self.cursor = self.connect.cursor()
        print("连接数据库成功")
    def process_item(self, item, spider):
        # 往数据库里面写入数据
        print("正在存入数据")
        self.cursor.execute(
            'insert into douban(name,href)VALUES ("{}","{}")'.format(item['title'], item['href']))
        self.connect.commit()
        return item

    # 关闭数据库
    def close_spider(self, spider):
        self.cursor.close()
        self.connect.close()
