import requests
import pandas as pd
from datetime import datetime
import json
import sys
# Falls das Modul lokal liegt, stelle sicher, dass der Pfad stimmt
from getsentimentFinBERT import get_finbert_score

BASE_URL = "https://api.queryly.com/cnbc/json.aspx?queryly_key=31a35d40a9a64ab3&query={ticker}&endindex={endindex}&batchsize=100&callback=&showfaceted=false&timezoneoffset=-120&facetedfields=formats&facetedkey=formats|&facetedvalue=!Press%20Release|&sort=date&additionalindexes=4cd6f71fbf22424d,937d600b0d0d4e23,3bfbe40caee7443e,626fdfcd96444f28"

UNIQUE_FIELD_JSON = 'url' 

ticker = "Micron"

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
    except (ValueError, TypeError):
        return None

def fetch_articles(endindex):
    url = BASE_URL.format(ticker = ticker, endindex=endindex)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status() 
        return response.json()
    except Exception as e:
        print(f"Error at API request {str(e)}")
        return None

def process_batch(data, existing_urls):
    if not data or 'results' not in data:
        return [], False

    articles = data['results']
    batch_data = []
    new_urls_in_batch = set()
    found_duplicate = False

    for article in articles:
        article_url = article.get(UNIQUE_FIELD_JSON)
        
        if article_url in existing_urls:
            found_duplicate = True
            continue
            
        try:
            title = article.get('cn:title', '').strip()
            description = article.get('description', '').strip()
            raw_date = article.get('datePublished')
            
            pub_date = parse_date(raw_date)
            input_text = f"{title}: {description}"
            
            # Sentiment analysis (FinBERT)
            sentiment_score = get_finbert_score(input_text)
            
            batch_data.append({
                "title": title,
                "description": description,
                "published_date": pub_date,
                "sentiment_score": sentiment_score,
                "url": article_url
            })
            new_urls_in_batch.add(article_url)

        except Exception as e:
            print(f"Error working on: {article_url}: {e}")

    return batch_data, found_duplicate

def get_news_dataframe(max_pages=5):
    all_articles_list = []
    seen_urls = set()
    endindex = 0
    batch_counter = 0

    print(f"Start fetching news for: ", ticker, " ...")

    while batch_counter < max_pages:
        batch_counter += 1
        print(f"Load batch {batch_counter} (Index: {endindex})...")

        data = fetch_articles(endindex)
        if not data or 'results' not in data:
            break

        current_batch, had_duplicates = process_batch(data, seen_urls)
        

        for art in current_batch:
            all_articles_list.append(art)
            seen_urls.add(art['url'])


        metadata = data.get('metadata', {})
        page_size = metadata.get('pagesize', 100)
        current_results_count = len(data.get('results', []))

        if current_results_count < page_size:
            print("Last page reached")
            break
            
        endindex += page_size


    df = pd.DataFrame(all_articles_list)
    
    if not df.empty:

        df['published_date'] = pd.to_datetime(df['published_date'])
        df = df.sort_values(by='published_date', ascending=False).reset_index(drop=True)
    
    print(f"Finished! {len(df)} articles processed")
    return df

if __name__ == "__main__":

    news_df = get_news_dataframe()
    path = "sentiment" + ticker + ".csv"
    news_df.to_csv(path, index="published_date")
    

    if not news_df.empty:
        print("\n--- Data preview ---")
        print(news_df[['published_date', 'title', 'sentiment_score']].head(10))


        

    else:
        print("Keine Daten gefunden.")