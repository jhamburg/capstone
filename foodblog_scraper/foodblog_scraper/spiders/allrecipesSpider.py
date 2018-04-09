import scrapy
import logging
import re
import numpy as np

class allrecipesSpider(scrapy.Spider):
    name = 'allrecipes'

    # recipe page has infinite scrolling but each scrolling loads a new page
    # manual testing found that pages >= 3350 don't exist so will loop to there
    start_urls = ['https://www.allrecipes.com/recipes/?page={}'.format(i) for i in range(3350)]
#     start_urls = ['https://www.allrecipes.com/recipes/?page={}'.format(i) for i in range(2)]
    
    def parse(self, response): 
        imageContainers = response.css("article.fixed-recipe-card div.grid-card-image-container")
        recipeLinks = imageContainers.xpath("./a[contains(@href, '/recipe')]")
        
        for recipeLink in recipeLinks:
            href = recipeLink.xpath("./@href").extract_first()
            yield response.follow(href, self.parse_recipe)
                
        
    def parse_recipe(self, response):
        """
        Actually scrape recipe information from the final recipe webpage
        """
               
        # Utility functions for extracting the time
        
        def clean_times(timeString, timeDict):
            "Reformats times to be consistent with food network scraper"
            
            byChar = ' '.join(re.split(r'(\d+|\w)', timeString)).strip()
            replaceFunc = lambda x: timeDict.get(x.group(), x.group())
            updList = [re.sub(r'(\w)', replaceFunc , byChar.lower())]
#             for time in timeList]
            return ' '.join(updList)

        
        def extract_times(timeSelector):
            "Returns a dictionary containing the different times (prep, cook, total)"
            
            res = {'total': [], 'prep': [], 'cook': []}
            
            # If time information doesn't exist
            if not timeSelector:
                return res
            
            timeTypes = timeSelector.xpath("./p/text()").extract()
            timeAmts = timeSelector.xpath("./time/@datetime").extract()
            
            timeStrings = [re.search(r'(\d\w+)', time)[0] for time in timeAmts]
            
            timeDict = {'h': 'hr', 'm': 'min', 'd': 'day', 's':'sec'}
            times = [clean_times(time, timeDict) for time in timeStrings]
            
            
            for timeType, time in zip(timeTypes, times):
                if timeType == 'Ready In':
                    timeType = 'total'
                res[timeType.lower()] = time
                
            return res
 
        # Get helper objects
        summarySelector = response.css('div.summary-background')
        ingredientsSelector = response.css('section.recipe-ingredients')
        directionsSelector = response.css('section.recipe-directions div.directions--section__steps')
                
        timeSelector = directionsSelector.css('ul.prepTime').xpath("./li[contains(.//p/@class, 'prepTime__item--type')]")
        times = extract_times(timeSelector)
        
        directions = directionsSelector.css('div.directions--section__steps ol.recipe-directions__list li span::text').extract()
        
        yield {
            'name': summarySelector.css('h1.recipe-summary__h1::text').extract_first(),
            'author': summarySelector.css('span.submitter__name::text').extract_first(),
            'totalTime': times['total'],
            'prepTime': times['prep'],
            'cookTime': times['cook'],
            'servings': '',
            'ingredients': ingredientsSelector.xpath("//span[contains(@class, 'recipe-ingred_txt') and ./@data-id]/text()").extract(),
            'tags': '',
            'directions': [txt.strip() for txt in directions]
        }