import streamlit as st
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType
import matplotlib.pyplot as plt

API_KEY = st.secrets["GNEWS_API_KEY"]
# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="News Sentiment Analyzer", layout="wide")
st.title("🗞️ Real-Time News Sentiment Analyzer")
st.write("Powered by GNews API, PySpark, and VADER")

# -----------------------------
# USER INPUT: API KEY
# -----------------------------
api_key = st.secrets["GNEWS_API_KEY"]

if api_key:
    try:
        # -----------------------------
        # FETCH NEWS
        # -----------------------------
        with st.spinner("Fetching top headlines..."):
            url = f'https://gnews.io/api/v4/top-headlines?lang=en&max=100&token={api_key}'
            response = requests.get(url)
            data = response.json()
            articles = data.get("articles", [])

            if not articles:
                st.error("No articles found. Please check your API key.")
                st.stop()

            # Convert to Pandas DataFrame
            df = pd.DataFrame(articles)[['title', 'description', 'publishedAt', 'url']]
            df.rename(columns={'publishedAt': 'published date'}, inplace=True)

            # -----------------------------
            # PYSPARK SESSION
            # -----------------------------
            spark = SparkSession.builder.appName("NewsSentiment").getOrCreate()
            spark_df = spark.createDataFrame(df)

            # -----------------------------
            # VADER Sentiment Analysis
            # -----------------------------
            analyzer = SentimentIntensityAnalyzer()

            def get_sentiment(text):
                if text:
                    return analyzer.polarity_scores(text)['compound']
                return 0.0

            def label_sentiment(score):
                if score >= 0.05:
                    return 'Positive'
                elif score <= -0.05:
                    return 'Negative'
                else:
                    return 'Neutral'

            # Register UDFs
            sentiment_udf = udf(get_sentiment, FloatType())
            label_udf = udf(label_sentiment, StringType())

            # Apply UDFs
            spark_df = spark_df.withColumn("sentiment", sentiment_udf(spark_df["title"]))
            spark_df = spark_df.withColumn("sentiment_label", label_udf(spark_df["sentiment"]))

            # Convert back to Pandas for Streamlit
            result_df = spark_df.toPandas()

            # -----------------------------
            # DISPLAY TABLE
            # -----------------------------
            st.subheader("📰 News with Sentiment Labels")
            st.dataframe(result_df[['title', 'sentiment_label', 'sentiment']], use_container_width=True)

            # -----------------------------
            # PLOT HISTOGRAM
            # -----------------------------
            st.subheader("📊 Sentiment Score Distribution")

            fig, ax = plt.subplots()
            ax.hist(result_df['sentiment'], bins=10, color='skyblue', edgecolor='black')
            ax.set_title("Sentiment Histogram")
            ax.set_xlabel("Sentiment Score")
            ax.set_ylabel("Number of Articles")
            st.pyplot(fig)

            # -----------------------------
            # PLOT PIE CHART
            # -----------------------------
            st.subheader("📈 Sentiment Category Breakdown")

            sentiment_counts = result_df['sentiment_label'].value_counts()
            st.write(sentiment_counts)

            fig2, ax2 = plt.subplots()
            ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'red', 'gray'])
            ax2.set_title("Sentiment Pie Chart")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred: {e}")
