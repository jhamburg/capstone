import scrapy
import logging
import re

class foodNetworkSpider(scrapy.Spider):
    name = 'foodNetwork_totalRecipes'
    
    start_urls = ['http://www.foodnetwork.com/topics/']
    
    def parse(self, response):
#         tmp = response.css('div.o-Capsule__m-Body li.m-PromoList__a-ListItem a::attr(href)')
#         tmp3 = tmp[:10]
        
#         for href in tmp3:
       for topicSelector in response.css('div.o-Capsule__m-Body li.m-PromoList__a-ListItem a'):
            topicName = topicSelector.xpath('./text()').extract_first()
        
            if topicName != 'Portobello Mushroom':
                href = topicSelector.css('a::attr(href)').extract_first()
                yield response.follow(href, self.parse_topic)
    
    def parse_topic(self, response):
        
        
        topicName = response.css('span.o-AssetTitle__a-HeadlineText::text').extract_first()
        numLinks = response.css('.o-SearchStatistics__a-MaxPage::text').extract_first()
        yield {
            'topic': topicName,
            'numLinks': re.search('(\d+)', numLinks).group(1)
        }
        
