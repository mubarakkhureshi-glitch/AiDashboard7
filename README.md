# AiDashboard7
from google.colab import drive
drive.mount('/content/drive')
# AI-Powered Market Insights Dashboard - Colab Version
# Using Kaggle Amazon Reviews dataset

# ------------------------
# Step 0: Install required packages
!pip install pandas numpy matplotlib seaborn nltk wordcloud plotly dash jupyter-dash

# ------------------------
# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import re
from collections import Counter
from io import BytesIO

# For displaying in Colab
from IPython.display import display, HTML

# ------------------------
# Step 2: Download NLTK VADER lexicon
nltk.download('vader_lexicon')

# ------------------------
# Step 3: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ------------------------
# Step 4: Load dataset files
folder_path = '/content/drive/MyDrive/amazon_reviews/*.csv'  # adjust if CSVs have a different extension
files = glob.glob(folder_path)

# Read and combine all files
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

# ------------------------
# Step 5: Keep only review text and rating
df = df[['review_body', 'star_rating']]
df.dropna(subset=['review_body'], inplace=True)

# ------------------------
# Step 6: Sentiment Analysis
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['review_body'].apply(get_sentiment)

# ------------------------
# Step 7: Word Cloud
text_all = ' '.join(df['review_body'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_all)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Amazon Reviews')
plt.show()

# ------------------------
# Step 8: Sentiment Distribution Chart
sentiment_counts = df['Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

fig_sentiment = px.bar(sentiment_counts, x='Sentiment', y='Count',
                       color='Sentiment',
                       title='Sentiment Distribution of Amazon Reviews')

# ------------------------
# Step 9: Rating Distribution
rating_counts = df['star_rating'].value_counts().sort_index().reset_index()
rating_counts.columns = ['Rating', 'Count']

fig_ratings = px.bar(rating_counts, x='Rating', y='Count',
                     title='Distribution of Star Ratings',
                     color='Rating', color_continuous_scale='Viridis')

# ------------------------
# Step 10: Top 20 Most Common Words
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

all_words = ' '.join(df['review_body'].apply(clean_text)).split()
counter = Counter(all_words)
most_common_words = counter.most_common(20)

common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
fig_words = px.bar(common_words_df, x='Word', y='Frequency',
                   title='Top 20 Most Common Words in Reviews')

# ------------------------
# Step 11: Sentiment by Rating
sentiment_by_rating = pd.crosstab(df['star_rating'], df['Sentiment'])
sentiment_by_rating_percent = sentiment_by_rating.div(sentiment_by_rating.sum(axis=1), axis=0) * 100

fig_sentiment_rating = px.imshow(sentiment_by_rating_percent,
                                 labels=dict(x="Sentiment", y="Star Rating", color="Percentage"),
                                 title="Sentiment Distribution by Star Rating",
                                 aspect="auto")
fig_sentiment_rating.update_xaxes(side="top")

# ------------------------
# Step 12: Create a simple dashboard display
display(HTML("<h1>AI-Powered Market Insights Dashboard</h1>"))
display(HTML("<h3>Amazon Product Reviews Analysis</h3>"))

# Display all visualizations
fig_sentiment.show()
fig_ratings.show()
fig_words.show()
fig_sentiment_rating.show()

# Display the top words table
display(HTML("<h3>Top 20 Most Common Words:</h3>"))
for word, freq in most_common_words:
    display(HTML(f"<p>{word}: {freq}</p>"))

# ------------------------
# Step 13: Additional Insights
# Calculate average rating by sentiment
avg_rating_by_sentiment = df.groupby('Sentiment')['star_rating'].mean().reset_index()
display(HTML("<h3>Average Star Rating by Sentiment:</h3>"))
for index, row in avg_rating_by_sentiment.iterrows():
    display(HTML(f"<p>{row['Sentiment']}: {row['star_rating']:.2f}</p>"))

# Calculate correlation between sentiment and rating
sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['sentiment_score'] = df['Sentiment'].map(sentiment_map)
correlation = df['sentiment_score'].corr(df['star_rating'])
display(HTML(f"<h3>Correlation between Sentiment and Star Rating: {correlation:.2f}</h3>"))
# Add this to the VERY END of your existing Colab code
import matplotlib.pyplot as plt

# Convert the entire notebook output to HTML
from IPython.display import HTML
import base64

# Create a simple HTML page with all your outputs
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Amazon Reviews Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; text-align: center; }
        .chart { margin: 30px 0; text-align: center; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI-Powered Market Insights Dashboard</h1>
        <h3>Amazon Product Reviews Analysis</h3>
"""

# Add your visualizations to the HTML
html_content += "<div class='chart'><h3>Word Cloud</h3>"

# Save word cloud to HTML
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Amazon Reviews')
plt.savefig('wordcloud.png', bbox_inches='tight', dpi=150)
plt.close()

# Read the image and convert to base64 for embedding
with open('wordcloud.png', 'rb') as f:
    wordcloud_base64 = base64.b64encode(f.read()).decode()
html_content += f'<img src="data:image/png;base64,{wordcloud_base64}" alt="Word Cloud">'
html_content += "</div>"

# Add other charts
html_content += "<div class='chart'><h3>Sentiment Distribution</h3>"
html_content += fig_sentiment.to_html(include_plotlyjs='cdn')
html_content += "</div>"

html_content += "<div class='chart'><h3>Rating Distribution</h3>"
html_content += fig_ratings.to_html(include_plotlyjs='cdn')
html_content += "</div>"

html_content += "<div class='chart'><h3>Most Common Words</h3>"
html_content += fig_words.to_html(include_plotlyjs='cdn')
html_content += "</div>"

html_content += "<div class='chart'><h3>Sentiment by Rating</h3>"
html_content += fig_sentiment_rating.to_html(include_plotlyjs='cdn')
html_content += "</div>"

# Add the text insights
html_content += "<div style='margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px;'>"
html_content += "<h3>Top 20 Most Common Words:</h3>"
for word, freq in most_common_words:
    html_content += f"<p><strong>{word}</strong>: {freq} occurrences</p>"

html_content += "<h3>Average Star Rating by Sentiment:</h3>"
for index, row in avg_rating_by_sentiment.iterrows():
    html_content += f"<p><strong>{row['Sentiment']}</strong>: {row['star_rating']:.2f} ⭐</p>"

html_content += f"<h3>Correlation between Sentiment and Star Rating: {correlation:.2f}</h3>"
html_content += "</div>"

html_content += """
    </div>
</body>
</html>
"""

# Save the HTML file
with open('dashboard.html', 'w') as f:
    f.write(html_content)

print("✅ Dashboard saved as 'dashboard.html'")
print("📁 Download this file and upload to GitHub as 'index.html'")
from google.colab import files
files.download('dashboard.html')
