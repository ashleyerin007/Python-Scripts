from serpapi import GoogleSearch
from newspaper import Article
from textblob import TextBlob
from datetime import datetime
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
import json
import time

# Config
API_KEY = "KEY"
TOPIC = "Topic"
MAX_RESULTS = 55
START_YEAR = datetime.now().year - 5

# Filtering
BLOCKED_DOMAINS = [
    #"proctoru.com",
    #"meazurelearning.com",
    "youtube.com",
    "facebook.com",
    "twitter.com"
]

def is_allowed_url(url):
    domain = urlparse(url).netloc.lower()
    return not any(blocked in domain for blocked in BLOCKED_DOMAINS)

# Search
def search_serpapi(topic):
    params = {
        "engine": "google",
        "q": topic,
        "num": MAX_RESULTS,
        "api_key": API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])
    return [r["link"] for r in results if "link" in r]

# Processing
def process_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        pub_date = article.publish_date
        if pub_date and pub_date.year < START_YEAR:
            return None  # Skip old content

        blob = TextBlob(article.text)
        sentiment = blob.sentiment.polarity
        word_freq = Counter(blob.word_counts).most_common(5)
        keywords = [word for word, _ in word_freq if len(word) > 3]

        return {
            "url": url,
            "title": article.title,
            "date": pub_date.strftime("%Y-%m-%d") if pub_date else None,
            "text": article.text[:500],
            "sentiment": sentiment,
            "keywords": ", ".join(keywords)
        }

    except Exception as e:
        print(f"[!] Failed to process {url}: {e}")
        return None

# Pipeline
urls = search_serpapi(TOPIC)
print(f"🔎 Found {len(urls)} URLs")

records = []
for url in urls:
    if not is_allowed_url(url):
        print(f"⛔ Skipped blocked domain: {url}")
        continue

    result = process_article(url)
    if result:
        records.append(result)
        print(f"✅ {result['title']} ({result['date']})")
    time.sleep(1)
  
df = pd.DataFrame(records)
df.to_csv("topic_articles.csv", index=False)
df.to_json("topic_articles.json", orient="records", indent=2)

print(f"\n✅ Saved {len(df)} articles to topic_articles.csv and topic_articles.json")
