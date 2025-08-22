# The Introduction of the MSc Project
## Overview
This project uses LLMs to perform sentiment analysis on financial news headlines to predict Bitcoin price trends. First we replicated FinBERT’s performance and retrained it on GDELT subset to improve its accuracy from 64.8% to 73.8%. Next, three sentiment scores were extracted from GDELT news dataset using retranied FinBERT model and the results were aggregated to develop multiple sentiment signals. Then we calculate Bitcoin returns from Bitcoin price dataset and construct multiple return signals. By calculating Pearson correlation coefficient, we find that the continuous sum sigmoid sentiment signal demonstrates the strongest correlation with Bitcoin returns. Based on this, we designed trading strategies, with one consistently outperforming the buy-and-hold benchmark by 20 percentage points, even during volatile and bearish periods.

## Questions
- Can LLM-based model improve prediction of Bitcoin price movements?

- Is the sentiment from news titles connected with Bitcoin returns?

- Can sentiment-based strategies outperform traditional trading strategies?

## Tools
LLM: FinBERT (base + retrained)

Libraries: Transformers, PyTorch, pandas, NumPy, yfinance

Others: GDELT 2.0 event database for news

## Data Preparation

We downloaded and input the news data into a dataframe. Cleaned and preprocessed the text with relabeling, lowecasing, removing stopwords and lemmatization.

```python
input_df = pd.read_csv('finbert_600.csv')
df = input_df[['title','ground_truth']]
df.columns = ['news','labels']
df['labels'].replace({1:2},inplace=True)
df['labels'].replace({0:1},inplace=True)
df['labels'].replace({-1:0},inplace=True)


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
df['news'] = df['news'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df["news"] = df["news"].str.lower()
df["news"] = df['news'].str.replace('[^\w\s]','')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
lemmatizer = WordNetLemmatizer()

#Function to apply for each word the proper lemmatization.
def lemmetize_titles(words):
    a = []
    tokens = word_tokenize(words)
    for token in tokens:
        lemmetized_word = lemmatizer.lemmatize(token)
        a.append(lemmetized_word)
    lemmatized_title = ' '.join(a)
    return lemmatized_title

df['lemmetized_titles'] = df['news'].apply(lemmetize_titles)
```

For the Bitcoin Price data, we downloaded using yfinance library, changed the data types and finally enriched the dataset with daily and weekly returns.

```python
raw_price_df = pd.read_csv('BTC-USD_price.csv')
raw_price_df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
raw_price_df = raw_price_df.iloc[2:].reset_index(drop=True)
price_df = raw_price_df[['Date','Close']]
price_df['Close'] = price_df['Close'].str.replace(',', '').astype(float)
price_df['1dayreturn'] = 0
for i in range(price_df['1dayreturn'].shape[0]-1):
    price_df['1dayreturn'][i] = (price_df['Close'][i+1] - price_df['Close'][i]) / price_df['Close'][i] * 100

price_df['1weekreturn'] = 0
for i in range(price_df['1weekreturn'].shape[0]-7):
    price_df['1weekreturn'][i] = (price_df['Close'][i+7] - price_df['Close'][i]) / price_df['Close'][i] * 100
price_df

```


## Analysis
1. FinBERT is retrained based on 600 manually labeled news headlines from GDELT 2.0 event database. The accuracy of the retrained FinBERT improved from 64.8% to 73.8%. ![retraining](retraining.png)

2. Using the retrained FinBERT model, the sentiment was extracted and 6 sentiment signals were built. Using the Bitcoin price data, 6 return signals were built as well. The best correlation exists between daily return signal and countinuous_sum_sigmoid sentiment signal.

3. Two different sentiment based strategies with different thresholds were backtested. The results show that the second sentiment strategy consistently outperforms the buy-and-hold benchmark by 20 percentage points, even during volatile and bearish periods![tradingsignal](sentiment3_2.png)


## Insights
### **1. Impact of LLM-Based Sentiment Analysis**
- LLM-based sentiment analysis provides **deeper market insights** by capturing complex contextual relationships from financial news.
- The **FinBERT model** demonstrates strong capability in extracting sentiment from Bitcoin-related news titles.
- However, to improve prediction accuracy, it is necessary to **retrain the FinBERT model** on more comprehensive datasets, such as the **GDELT 2.0 event database**, to capture domain-specific nuances.

### **2. Relationship Between Sentiment and Bitcoin Returns**
- There is a clear correlation between **market sentiment** derived from news and **Bitcoin daily returns**.
- Among multiple sentiment-based signals, the **`continuous_sum_sigmoid`** signal shows the **highest correlation** with Bitcoin price changes.
- This signal effectively identifies potential turning points in the market and can be used as an **indicator** for when to buy or sell Bitcoin.

### **3. Performance of Sentiment-Based Trading Strategies**
- **Sentiment-driven trading strategy** consistently achieves **higher positive returns** compared to a **baseline buy-and-hold strategy**.
- Backtesting shows that sentiment signals enable **timely entry and exit points**, reducing exposure to market downturns while maximizing profit opportunities.
- By leveraging news-driven sentiment, trading decisions become **more dynamic** and **data-driven**.

### **4. Enhancing Trading Signal Performance**
- While FinBERT-based sentiment signals provide strong predictive power, **advanced machine learning techniques** could further enhance accuracy.
- Approaches such as:
  - **Reinforcement learning** for optimizing trading decisions.
  - **Hybrid sentiment-price models** combining FinBERT outputs with historical price data.
  - **Time-series transformers** for capturing long-term dependencies in sentiment and price movements.
- These enhancements could lead to **better risk-adjusted returns** and more robust trading strategies.
