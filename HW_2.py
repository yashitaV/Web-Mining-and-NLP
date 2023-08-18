#!/usr/bin/env python
# coding: utf-8

# <h1><center>HW2: Scrape Hotel Reviews</center></h1>

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. </div>

# Choose one hotel at tripadvisor.com (e.g.https://www.tripadvisor.com/Hotel_Review-g60763-d23448880-Reviews-Motto_by_Hilton_New_York_City_Chelsea-New_York_City_New_York.html) to scrape reviews

# ## Q1. Scrape the first page
# 
# Write a function `getReviews(page_url, driver)` to scrape all **reviews on the first page**, which
# - accepts two input parameters:
#     - `page_url` is the URL to be scraped
#     - `driver` is the selenium web driver object initiated. Firefox, Chrome, or any web driver is acceptable.
# - locates all the reviews on this page, and for each review, scrape the following 
#     - `username` (see (1) in Figure)
#     - `helpful votes` (see (2) in Figure)
#     - `rating` (see (3) in Figure)
#     - `title` (see (4) in Figure)
#     - `review` (see (5) in Figure. You can just scrape the truncated text. You don't have to expand it.
#     - `date of stay` (see (6) in Figure)
# - if a field, e.g., rating, is missing, uses `None` to indicate it. 
# - collects all reviews as a DataFrame and each field as a column
# - returns the dataframe

# ## Q2 (bonus): Scrape full text in more reviews
# 
# Modify the function you defined in Q1 as `getReviews(page_url, driver, expand_review = False, limit = 10)` as follows:
# - If `expand_review = True`, click the "read_more" button (see (7) in Figure) of each review and collect the full text.
# - Collect the required number of reviews as specified by `limit`. For example, if `limit=50`, you'll need to collect at least 50 reviews. This requires you to keep clicking the `next` button (see (8) in Figure), if clickable, until the required number of reviews are collected.
# - Return a dataframe as specified in Q1.

# ![alt text](tripadvisor.png "TripAdvisor")

# ## Solution

# In[5]:


import requests
from bs4 import BeautifulSoup  
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re


# ### Q1

# In[76]:


def getReviews(page_url, driver):

    result = None
    
    # add your code here
    
    
    return result


# In[77]:


# page_url = "https://www.tripadvisor.com/Hotel_Review-g60763-d23448880-Reviews-or10-Motto_by_Hilton_New_York_City_Chelsea-New_York_City_New_York.html#REVIEWS"

executable_path = '../Notes/Web_Scraping/driver/geckodriver'
driver = webdriver.Firefox(executable_path=executable_path)

reviews = getReviews(page_url, driver)

driver.quit()

reviews


# In[37]:


import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

def getReviews(page_url, driver):
    # Open the page
    driver.get(page_url)

    # Wait for the page to load
    time.sleep(5)

    # Locate all the reviews on the page
    reviews = driver.find_elements(By.XPATH, "//div[@data-test-target='reviews-tab']/div")

    # Create an empty list to store the extracted information
    review_list = []

    # Extract the information for each review
    for review in reviews:
        # Extract the username
        username_element = review.find_element(By.XPATH, './/div[contains(@class="ui_header_link.uyyBf"]')
        username = username_element.text if username_element else None

        # Extract the helpful votes
        helpful_element = review.find_element(By.XPATH, './/div[@class="MziKN"]')
        helpful_text = helpful_element.text if helpful_element else ''
        helpful_votes = int(helpful_text.split()[0]) if helpful_text else None

        # Extract the rating
        rating_element = review.find_element(By.XPATH, './/span[contains(@class, "Hlmiy.F1")]')
        rating = int(rating_element.get_attribute('class').split('_')[-1]) if rating_element else None

        # Extract the title
        title_element = review.find_element(By.XPATH, './/div[@class="Qwuub"]/a')
        title = title_element.text if title_element else None

        # Extract the review  
        review_element = review.find_element(By.XPATH, './/span[contains(@class="QewHA.H4._a"]')
        review = review_element.text if review_element else None

        # Extract the date of stay
        stay_element = review.find_element(By.XPATH, './/div[contains(@class, "teHYY._R.Me.S4.H3")]')
        stay = stay_element.text if stay_element else None

        # Create a dictionary to store the extracted information
        review_dict = {'username': username,
                       'helpful_votes': helpful_votes,
                       'rating': rating,
                       'title': title,
                       'review': review,
                       'stay': stay}

        # Append the dictionary to the list
        review_list.append(review_dict)

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(review_list)

    return df


# In[39]:


driver = webdriver.Safari()

# Scrape the first page of reviews for a hotel
page_url = "https://www.tripadvisor.com/Hotel_Review-g60763-d23448880-Reviews-or10-Motto_by_Hilton_New_York_City_Chelsea-New_York_City_New_York.html#REVIEWS"

driver.get("https://www.tripadvisor.com/Hotel_Review-g60763-d23448880-Reviews-or10-Motto_by_Hilton_New_York_City_Chelsea-New_York_City_New_York.html#REVIEWS"
)
getReviews(page_url, driver)


# In[38]:


driver.quit()


# ### Q2. Bonus question

# In[ ]:


def getReviews(page_url, driver, expand_review = False, limit = 10):

    result = None
    
    # add your code here
    
    
    return result


# In[78]:


# Test

page_url = "https://www.tripadvisor.com/Hotel_Review-g60763-d23448880-Reviews-or10-Motto_by_Hilton_New_York_City_Chelsea-New_York_City_New_York.html#REVIEWS"

executable_path = '../Notes/Web_Scraping/driver/geckodriver'
driver = webdriver.Firefox(executable_path=executable_path)

reviews = getReviews(page_url, driver, expand_review = True, limit = 50)

driver.quit()

reviews


# ### Check if full text is collected

# In[79]:


# check if full text is collected

reviews.text.iloc[1]


# ## Main block for testing

# In[ ]:


# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":  
    
    # Test Question 1

    print("\nTest Question 1")
    page_url = "https://www.tripadvisor.com/Hotel_Review-g60763-d23448880-Reviews-or10-Motto_by_Hilton_New_York_City_Chelsea-New_York_City_New_York.html#REVIEWS"

    executable_path = '../Notes/Web_Scraping/driver/geckodriver'
    driver = webdriver.Firefox(executable_path=executable_path)

    reviews = getReviews(page_url, driver)

    driver.quit()

    print(reviews.head())
    
    
    #3 Test Question 2
    
    print("\nTest Question 2")
    page_url = "https://www.tripadvisor.com/Hotel_Review-g60763-d23448880-Reviews-or10-Motto_by_Hilton_New_York_City_Chelsea-New_York_City_New_York.html#REVIEWS"

    executable_path = '../Notes/Web_Scraping/driver/geckodriver'
    driver = webdriver.Firefox(executable_path=executable_path)

    reviews = getReviews(page_url, driver, expand_review = True, limit = 50)

    driver.quit()

    print(reviews.head())
    

