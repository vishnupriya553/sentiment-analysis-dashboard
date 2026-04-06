import streamlit as st
from model import predict_sentiment
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Page config
st.set_page_config(page_title="Sentiment Dashboard", page_icon="📊")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

st.write("### Enter text to analyze sentiment")

# Input box
user_input = st.text_input("")

if st.button("Analyze"):
    result = predict_sentiment(user_input)
    
    if result == "Positive":
        st.success(f"Sentiment: {result} 😊")
    elif result == "Negative":
        st.error(f"Sentiment: {result} 😠")
    else:
        st.warning(f"Sentiment: {result} 😐")

# Load dataset
df = pd.read_csv("data.csv")

# Sentiment distribution
st.subheader("Sentiment Distribution")
counts = df["Sentiment"].value_counts()

fig, ax = plt.subplots()
ax.bar(counts.index, counts.values)
st.pyplot(fig)

# Pie chart
st.subheader("Sentiment Pie Chart")
fig2, ax2 = plt.subplots()
ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
st.pyplot(fig2)

# Word cloud
st.subheader("Word Cloud")
text = " ".join(df["Text"])
wordcloud = WordCloud(background_color='white').generate(text)

fig3, ax3 = plt.subplots()
ax3.imshow(wordcloud)
ax3.axis("off")
st.pyplot(fig3)