# -*- coding: utf-8 -*-
import scrapy
#导入几个包
from scrapy.http import Request
from scrapy.selector import Selector
from scrapy.spiders import Rule
from tianya.items import TianyaItem
from bs4 import BeautifulSoup

global i
i = 0
class TiantaSpider(scrapy.Spider):
    name = 'tianya'
    allowed_domains = ['tianya.com']
    #自己替换start_urls
    start_urls = ['https://movie.douban.com/top250']

    def parse(self, response):
        #选择器
        selector=Selector(response)
        #拿东西
        coments=selector.xpath('//div[@class="info"]')
        #print(coments)
        for eachone in coments:
            #print(eachone)
            title=eachone.xpath('div[@class="hd"]/a/span[@class="title"]/text()').extract()[0]
            href = eachone.xpath('div[@class="hd"]/a/@href').extract()[0]
            item=TianyaItem()
            item['title']=title
            item['href']=href
            print(title,href)
            yield item
            #yield Request(response.urljoin(href),callback=self.parse_namedetail)
        for i in range(1,11):
                nextlink = 'https://movie.douban.com/top250?start='+str(i*25)+'&filter='
                nextlink=response.urljoin(nextlink)
                print(i*25)
                yield Request(nextlink,callback=self.pa,dont_filter=True)
    def pa(self, response):
        # 选择器
        selector = Selector(response)
        # 拿东西
        coments = selector.xpath('//div[@class="info"]')
        # print(coments)
        for eachone in coments:
            # print(eachone)
            title = eachone.xpath('div[@class="hd"]/a/span[@class="title"]/text()').extract()[0]
            href = eachone.xpath('div[@class="hd"]/a/@href').extract()[0]
            item = TianyaItem()
            item['title'] = title
            item['href'] = href
            print(title, href)
            yield item
            yield Request(response.urljoin(href), callback=self.parse_namedetail)
        #return item
    def parse_namedetail(self,response):
        selector=Selector(response)
        desc=selector.xpath('//div[@id="link-report"]/span/text()')
        item=response.meta('name')
        item['desc']=desc
        print(')))))')
        #yield item