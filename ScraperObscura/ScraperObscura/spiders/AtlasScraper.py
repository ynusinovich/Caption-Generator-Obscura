# -*- coding: utf-8 -*-
import scrapy
import time

class AtlasscraperSpider(scrapy.Spider):
    name = 'AtlasScraper'
    start_urls = ['https://www.atlasobscura.com/places/']
    page_num = 2

    def parse(self, response):
        image_urls = response.xpath("//div/a/figure/img/@data-src").extract()
        image_descriptions = response.css(".js-subtitle-content::text").extract()
        num_pages = int(response.xpath("//span[@class='last']/a/@href").extract()[0].split("=")[1])

        for item in zip(image_urls, image_descriptions):
            image_dict = {
                "url": item[0],
                "description": item[1],
                "response": response.status
            }
            time.sleep(5)
            yield(image_dict)

        next_page = "https://www.atlasobscura.com/places?page=" + str(AtlasscraperSpider.page_num)
        if AtlasscraperSpider.page_num <= num_pages:
            yield(response.follow(next_page, callback = self.parse))
            AtlasscraperSpider.page_num += 1
