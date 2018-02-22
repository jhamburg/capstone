import scrapy
import logging
import re

class foodNetworkSpider(scrapy.Spider):
    name = 'foodNetwork'
    allowed_domains = ['https://www.foodnetwork.com/']
    start_urls = ['http://www.foodnetwork.com/topics/']
    
    def parse(self, response):       
        topicSelectors = response.css('div.o-Capsule__m-Body li.m-PromoList__a-ListItem a')
        topicSelectors = topicSelectors[:10]
        
        for topicSelector in topicSelectors:
            topicName = topicSelector.xpath('./text()').extract_first()
        
            # Portobello Mushroom has an issue --- contains over 100,000 recipes and 
            # most are not mushroom recipesf
            if topicName != 'Portobello Mushroom':
                href = topicSelector.css('a::attr(href)').extract_first()
                yield response.follow(href, self.parse_topic)
                
    
    def parse_topic(self, response):
        """
        On the topic page, navigate to actual recipe or if it is a video,
        navigate to the video page and then to the recipe page.
        
        Will also navigate the next button while it exists.
        """
        # follow links to individual recipes. Specifically only chooses recipeResults
        # since other results may also exist (videos/article)
        
        recipeResults = response.css('.o-RecipeResult')
        videoResults = response.css('.o-SingleVideoResult')
        
        def getLinks(resultSelectors):
            return(resultSelectors.xpath('.//h3[@class="m-MediaBlock__a-Headline"]/a/@href'))
        
        for href in getLinks(recipeResults):
            yield response.follow(href, self.parse_recipe)
        
        for href in getLinks(videoResults):
            yield response.follow(href, self.parse_video_page)
        
#             # Topic link may go to a page with a video of the directions instead
#             # of the actual recipe. Need to continue past this middle page

#             if (href.extract().find('video') > 0):
#                 yield response.follow(href, self.parse_video_page) 
        
        # follow pagination links
        pageLink = response.css('a.o-Pagination__a-NextButton::attr(href)').extract_first()
        
        if (pageLink.find('www') > 0):
            yield response.follow(pageLink, self.parse_topic)
            
    def parse_video_page(self, response):
        """
        Continue on to recipe page if possible
        """
        # continue to actual recipe
        videoButtonLink = response.css('a.o-VideoMetadata__a-Button')
        buttonText = videoButtonLink.xpath('./text()').extract_first()
        
        # Run if button actually exists
        if buttonText:
            # Makes sure button is to a recipe and not another link
            if buttonText.lower().find('recipe') > 0:
                recipeLink = videoButtonLink.xpath('./@href').extract_first()
                yield response.follow(recipeLink, self.parse_recipe)
        
    def parse_recipe(self, response):
        """
        Actually scrape recipe information from the final recipe webpage
        """
        
        # Utility function for extracting first string and cleaning it
        def extract_with_xpath(selector, query):
            res = selector.xpath(query).extract_first()            
            if res:
                return(res.strip())
            return(None)
        
        # Utility function for extracting the time
        def extract_time(timeType, opts, vals): 
            # In case there isn't a time presented
            if not opts:
                return(None)
            
            # Remove punctuation
            opts = [re.sub(r'\W+', '', txt.lower()) for txt in opts]
            
            # Will check for keyword "active" as well or will return None
            if not timeType in opts:
                if timeType == 'prep':
                    timeType = 'active'
                    if not timeType in opts:
                        return(None)
                else:
                    return(None)
            
            # Use pop to get the string instead of a list
            res = [i.strip() for i,j in zip(vals, opts) if j == timeType]
            return(res.pop())
 
        # Get helper objects
        attributionSelector = response.xpath('//*[@data-module="recipe-lead"]')
        directions = response.xpath("//div[@class and contains(concat(' ', normalize-space(@class), ' '), ' method parbase section ')]//div[@class = 'o-Method__m-Body']/p/text()").extract()
        tags = response.xpath("//div[@class and contains(concat(' ', normalize-space(@class), ' '), ' parbase section tags')]").css('a.a-Tag::text').extract()
        timeDiv = attributionSelector.css('.o-Time')
        if timeDiv:
            timeDiv = timeDiv[0]
        timeOpts = timeDiv.xpath('.//dt/text()').extract()
        timeVals = timeDiv.xpath('.//dd/text()').extract()
        
        yield {
            'name': extract_with_xpath(attributionSelector, './/*[@class="o-AssetTitle__a-HeadlineText"]/text()'),
            'author': extract_with_xpath(attributionSelector, './/span[@class="o-Attribution__a-Name"]//a/text()'),
            'totalTime': extract_time('total', timeOpts, timeVals),
            'prepTime': extract_time('prep', timeOpts, timeVals),
            'cookTime': extract_time('cook', timeOpts, timeVals),
            'servings': extract_with_xpath(attributionSelector, './/*[@class="o-RecipeInfo__a-Description"]/text()'),
            'ingredients': response.css('.o-Ingredients__a-ListItem label::text').extract(),
            'tags': [tag.strip() for tag in tags],
            'directions': [txt.strip() for txt in directions]
        }