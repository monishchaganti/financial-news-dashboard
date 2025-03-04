import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Set page title and layout
st.set_page_config(page_title="Financial News Sentiment Analysis", layout="wide")

# Load the cleaned data with VADER sentiment scores
df = pd.read_csv('cleaned_financial_news.csv')

# Title of the dashboard
st.title("Financial News Sentiment Analysis Dashboard")

# Sidebar for user input
st.sidebar.header("Filters")
selected_media = st.sidebar.selectbox("Select Media Source", df['media'].unique())

# Filter data based on selected media source
filtered_df = df[df['media'] == selected_media]

# Display filtered data
st.subheader(f"News Articles from {selected_media}")
st.write(filtered_df[['title', 'vader_title_sentiment_label']])

# 1. Sentiment Distribution
st.subheader("Sentiment Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='vader_title_sentiment_label', data=filtered_df, palette='Set2', order=['Positive', 'Neutral', 'Negative'], ax=ax)
plt.xlabel("Sentiment")
plt.ylabel("Count")
st.pyplot(fig)

# 2. Top Keywords
st.subheader("Top Keywords")
all_text = ' '.join(filtered_df['cleaned_title'] + ' ' + filtered_df['cleaned_desc'])
word_freq = Counter(all_text.split())
top_20_words = word_freq.most_common(20)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=[word[1] for word in top_20_words], y=[word[0] for word in top_20_words], palette='viridis', ax=ax)
plt.xlabel("Frequency")
plt.ylabel("Words")
st.pyplot(fig)

# 3. Word Cloud
st.subheader("Word Cloud")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
fig, ax = plt.subplots(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig)

# 4. Overall Sentiment Hypothesis
overall_sentiment = filtered_df['vader_title_sentiment'].mean()
if overall_sentiment > 0.05:
    sentiment_impact = "The overall sentiment is positive, which could indicate a bullish market sentiment."
elif overall_sentiment < -0.05:
    sentiment_impact = "The overall sentiment is negative, which could indicate a bearish market sentiment."
else:
    sentiment_impact = "The overall sentiment is neutral, suggesting no strong market direction."

st.subheader("Overall Sentiment Analysis")
st.write(f"Overall Sentiment Score: {overall_sentiment:.2f}")
st.write(f"Hypothesis: {sentiment_impact}")