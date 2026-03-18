"""
Crawler to extract reviews from coffeereview.com
"""
import scrapy


class CoffeeSpider(scrapy.Spider):
  name = "coffee"
  start_urls = [f"https://www.coffeereview.com/review/"]
  max_pages = 447

  def parse(self, response):
    url = response.url
    current_page = int(url.split('/page/')[-1].strip('/')) if '/page/' in url else 1

    # Enter each Coffee Review page
    coffee_page_links = response.css("h2.review-title a")
    yield from response.follow_all(coffee_page_links, self.parse_coffee)

    # Enter each pagination
    if current_page <= self.max_pages:
      pagination_links = response.css("li.pagination-next a")
      yield from response.follow_all(pagination_links, self.parse)

  def parse_coffee(self, response):
    # Repeated headers
    attributes = [f"div.column.col-{i} table.review-template-table tr" for i in range(1, 3)]

    # Coffee attributes
    attributes_table = {}
    for attribute in attributes:
      for row in response.css(attribute):
        label = row.css("td:nth-child(1)::text").get()
        value = row.css("td:nth-child(2)::text").get()
        if label and value:
          attributes_table[label.strip().rstrip(":")] = value.strip()
    
    # Open text: reviews
    reviews = {}
    h2_tags = response.xpath("//div[@class='review-template']//h2")

    for i, h2 in enumerate(h2_tags):
      key = h2.xpath("string()").get().strip()
      # get all following p tags, then stop at the next h2 using position
      if i < len(h2_tags) - 1:
        next_h2 = h2_tags[i + 1]
        paragraphs = h2.xpath(
          "following-sibling::p[count(preceding-sibling::h2) = %d]" % (i + 1)
        )
      else:
        paragraphs = h2.xpath(
          "following-sibling::p[count(preceding-sibling::h2) = %d]" % (i + 1)
        )
      value = " ".join(p.xpath("string()").get().strip() for p in paragraphs)
      reviews[key] = value

    yield {
      "rating": response.css("span.review-template-rating::text").get(),
      "roaster": response.css("p.review-roaster::text").get(),
      "bean": response.css("h1.review-title::text").get(),
      **attributes_table,
      **reviews
    }
