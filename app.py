import streamlit as st
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType
import matplotlib.pyplot as plt

st.set_page_config(page_title="News Sentiment Analyzer", layout="wide")
st.title("🗞️ Real-Time News Sentiment Analyzer")
st.write("Powered by GNews API, PySpark, and VADER")

# Load API key from Streamlit secrets
API_KEY = st.secrets["GNEWS_API_KEY"]

if not API_KEY:
    st.error("API key not found! Please add your GNEWS_API_KEY to Streamlit secrets.")
    st.stop()

try:
    # Fetch news articles
    url = f'https://gnews.io/api/v4/top-headlines?lang=en&max=100&token={API_KEY}'
    response = requests.get(url)
    data = response.json()
    articles = data.get("articles", [])

    if not articles:
        st.error("No articles found. Please check your API key and API usage limits.")
        st.stop()

    # Prepare DataFrame
    df = pd.DataFrame(articles)[['title', 'description', 'publishedAt', 'url']]
    df.rename(columns={'publishedAt': 'published date'}, inplace=True)

    # Initialize Spark Session
    spark = SparkSession.builder.appName("NewsSentiment").getOrCreate()
    spark_df = spark.createDataFrame(df)

    # VADER Sentiment Analyzer
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        if text:
            return float(analyzer.polarity_scores(text)['compound'])
        return 0.0

    def label_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    sentiment_udf = udf(get_sentiment, FloatType())
    label_udf = udf(label_sentiment, StringType())

    spark_df = spark_df.withColumn("sentiment", sentiment_udf(spark_df["title"]))
    spark_df = spark_df.withColumn("sentiment_label", label_udf(spark_df["sentiment"]))

    # Convert to Pandas for display and plotting
    result_df = spark_df.toPandas()

    # Display news with sentiment
    st.subheader("📰 News Headlines with Sentiment Labels")
    st.dataframe(result_df[['title', 'sentiment_label', 'sentiment']], use_container_width=True)

    # Plot sentiment distribution histogram
    st.subheader("📊 Sentiment Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(result_df['sentiment'], bins=10, color='skyblue', edgecolor='black')
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Sentiment Histogram")
    st.pyplot(fig)

    # Pie chart of sentiment labels
    st.subheader("📈 Sentiment Category Breakdown")
    sentiment_counts = result_df['sentiment_label'].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=['green', 'red', 'gray'])
    ax2.set_title("Sentiment Pie Chart")
    st.pyplot(fig2)

except Exception as e:
    st.error(f"An error occurred: {e}")
