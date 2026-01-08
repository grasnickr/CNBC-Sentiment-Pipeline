# CNBC Sentiment  Pipeline

An automated pipeline designed to extract and analyze financial news sentiment. This project combines **reverse-engineered API access** with state-of-the-art **Natural Language Processing (NLP)** to quantify market sentiment for specific stock tickers.


---

## 🚀 Features

* **Undocumented API Integration**: Efficiently fetches news data directly via CNBC’s internal search infrastructure.
* **Financial Sentiment Analysis**: Utilizes the **ProsusAI/FinBERT** model, specifically fine-tuned for financial texts.
* **Pandas Workflow**: Automatically generates cleaned DataFrames ready for downstream analysis or machine learning models like LSTMs.
* **Hardware Optimized**: Supports CUDA acceleration for fast inference on NVIDIA GPUs.

---

## 🛠 Installation

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/CNBC-Sentiment-Pipeline.git](https://github.com/your-username/CNBC-Sentiment-Pipeline.git)
    cd CNBC-Sentiment-Pipeline
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📈 Usage (Library Import)

The logic is modular and can be imported into research scripts or automated trading strategies.

```python
from StockSentiment import get_news_sentiment

# Parameters: maxpages (100 articles per page), ticker symbol
ticker = "AAPL"
df = get_news_sentiment(maxpages=5, ticker=ticker)

# Preview analyzed data
if not df.empty:
    print(df[['published_date', 'title', 'sentiment_score']].head())