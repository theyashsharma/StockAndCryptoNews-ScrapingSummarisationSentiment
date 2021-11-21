#importing baseline dependencies
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import re
import csv

#setting up summarization model
model_name = "human-centered-summarization/financial-summarization-pegasus" #using financial-summarization-pegasus model from huggingface transformers
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

#considering two stocks for summarisation
monitored_tickers = ['ZOMATO', 'NYKAA']

#searching for stock news by moneycontrol on google
def search_for_stock_news_urls(ticker):
    search_url = "https://www.google.com/search?q=money+control+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a') #finding all anchor tags
    hrefs = [link['href'] for link in atags] #taking href anchor tags further
    return hrefs 

raw_urls = {ticker:search_for_stock_news_urls(ticker) for ticker in monitored_tickers}

#stripping or cleaning the URLs
exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']

def strip_unwanted_urls(urls, exclude_list):
  val = []
  for url in urls:
    if 'https://' in url and not any (exclude_word in url for exclude_word in exclude_list): #excluding the links that contain words in 'exclude_list'
      res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
      val.append(res)
  return list(set(val))

cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}

#scraping the news from cleaned URLs
def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs: #looping through all cleaned URLs
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find(class_ = 'page_left_wrapper').find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350] #only grabbing first 350 words because summarization model have some limit
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES

articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}

#summarizing the scraped news from all URLs
def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt') #encoding articles
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True) #generating summaries in encoded form
        summary = tokenizer.decode(output[0], skip_special_tokens=True) #decoding the generated summaries
        summaries.append(summary)
    return summaries

summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}

#applying sentiment analysis on summaries to find positive and negative scores
sentiment = pipeline('sentiment-analysis') #importing sentiment-analysis from transformer pipeline
scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers} #calculating scores

#creating output array of derived sentiments and summaries
def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]
            ]
            output.append(output_this)
    return output

final_output = create_output_array(summaries, scores, cleaned_urls)
final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Confidence', 'URL'])

#exporting the array into csv
with open('assetsummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)
