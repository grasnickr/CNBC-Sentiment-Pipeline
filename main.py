import argparse
from StockSentiment import get_news_sentiment

def main():
    parser = argparse.ArgumentParser(description="CNBC Sentiment Analysis Tool")
    
    # Args
    parser.add_argument("--ticker", type=str, help="Ticker symbol of the desired stock")
    parser.add_argument("--pages", type=int, default=5, help="Number of Pages. 1 Page = 100 articles")
    
    args = parser.parse_args()

    pipeline = get_news_sentiment(ticker=args.ticker)
    df = pipeline.run(max_pages=args.pages)
    print(f"Erfolgreich {len(df)} Artikel für {args.ticker} verarbeitet.")

if __name__ == "__main__":
    main()