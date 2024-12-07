import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
from textblob import TextBlob
from tqdm import tqdm

def load_data(filename):
    """Load and preprocess the data from text file."""
    reviews = []
    labels = []
    
    # Count total lines first for progress bar
    with open(filename, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # Read with progress bar
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Loading data"):
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                label = 1 if parts[0] == '__label__2' else 0
                review = parts[1]
                labels.append(label)
                reviews.append(review)
    
    return pd.DataFrame({'review_text': reviews, 'sentiment': labels})

def get_summary_statistics(df):
    """Calculate and print summary statistics."""
    total_reviews = len(df)
    positive_reviews = df['sentiment'].sum()
    negative_reviews = total_reviews - positive_reviews
    
    print("\nSummary Statistics:")
    print(f"Total number of reviews: {total_reviews}")
    print(f"Positive reviews: {positive_reviews} ({positive_reviews/total_reviews*100:.2f}%)")
    print(f"Negative reviews: {negative_reviews} ({negative_reviews/total_reviews*100:.2f}%)")

def get_advanced_statistics(df):
    """Calculate and print advanced statistics about the reviews."""
    # Average review length
    df['review_length'] = df['review_text'].str.len()
    df['word_count'] = df['review_text'].str.split().str.len()
    
    print("\nAdvanced Statistics:")
    print(f"Average review length (characters): {df['review_length'].mean():.2f}")
    print(f"Average word count: {df['word_count'].mean():.2f}")
    print(f"Longest review: {df['review_length'].max()} characters")
    print(f"Shortest review: {df['review_length'].min()} characters")

def generate_word_clouds(df):
    """Generate word clouds for positive and negative reviews."""
    sample_size = 1000  # Reduced from 10000
    df_sample = df.sample(n=sample_size)
    
    plt.figure(figsize=(15, 6))
    
    stop_words = set(['the', 'and', 'is', 'in', 'it', 'to', 'that', 'this', 'was', 'for'])
    
    def preprocess_text(text_series):
        text = ' '.join(text_series).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove stop words
        words = text.split()
        return ' '.join(word for word in words if word not in stop_words)
    
    # Positive reviews word cloud
    positive_text = preprocess_text(df_sample[df_sample['sentiment'] == 1]['review_text'])
    positive_wordcloud = WordCloud(width=400, height=200, 
                                 background_color='white',
                                 max_words=100).generate(positive_text)  # Limit words
    
    plt.subplot(1, 2, 1)
    plt.imshow(positive_wordcloud)
    plt.title('Common Words in Positive Reviews')
    plt.axis('off')
    
    # Negative reviews word cloud
    negative_text = preprocess_text(df_sample[df_sample['sentiment'] == 0]['review_text'])
    negative_wordcloud = WordCloud(width=400, height=200,  
                                 background_color='white',
                                 max_words=100).generate(negative_text)  
    
    plt.subplot(1, 2, 2)
    plt.imshow(negative_wordcloud)
    plt.title('Common Words in Negative Reviews')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_common_phrases(df, n=10):
    """Analyze most common phrases in positive and negative reviews."""
    # Reduce sample size
    sample_size = 1000  
    df_sample = df.sample(n=sample_size)
    
    def get_bigrams(text):
        # Preprocess text first
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        # Filter out common words
        stop_words = set(['the', 'and', 'is', 'in', 'it', 'to', 'that', 'this', 'was', 'for'])
        words = [word for word in words if word not in stop_words]
        return [' '.join(pair) for pair in zip(words[:-1], words[1:])]
    
    positive_bigrams = []
    negative_bigrams = []
    
    # Use vectorized operations instead of loop
    positive_reviews = df_sample[df_sample['sentiment'] == 1]['review_text']
    negative_reviews = df_sample[df_sample['sentiment'] == 0]['review_text']
    
    positive_bigrams = [bigram for text in positive_reviews for bigram in get_bigrams(text)]
    negative_bigrams = [bigram for text in negative_reviews for bigram in get_bigrams(text)]
    
    print("\nMost Common Phrases:")
    print("\nPositive Reviews:")
    for phrase, count in Counter(positive_bigrams).most_common(n):
        print(f"'{phrase}': {count} times")
    
    print("\nNegative Reviews:")
    for phrase, count in Counter(negative_bigrams).most_common(n):
        print(f"'{phrase}': {count} times")

def plot_review_length_distribution(df):
    """Plot the distribution of review lengths."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='review_length', hue='sentiment', bins=50)
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Number of Characters')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='word_count', hue='sentiment', bins=50)
    plt.title('Distribution of Word Counts')
    plt.xlabel('Number of Words')
    
    plt.tight_layout()
    plt.show()

def plot_sentiment_distribution(df):
    """Plot the distribution of sentiments."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sentiment')
    plt.title('Distribution of Review Sentiments')
    plt.xlabel('Sentiment (0: Negative, 1: Positive)')
    plt.ylabel('Count')
    plt.show()

def train_model(train_df):
    """Train a model to predict sentiments."""
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(train_df['review_text'])
    y = train_df['sentiment']
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    val_pred = model.predict(X_val)
    print("\nModel Performance:")
    print(classification_report(y_val, val_pred))
    
    return model, vectorizer

def main():
    # Load training data
    print("Loading training data...")
    train_df = load_data('train.ft.txt')
    
    # Get summary statistics
    get_summary_statistics(train_df)
    get_advanced_statistics(train_df)
    
    # Ask user what analysis they want to run
    print("\nWhat analysis would you like to run?")
    print("1. Generate word clouds")
    print("2. Analyze common phrases")
    print("3. Plot distributions")
    print("4. Train model")
    print("5. Run all analyses")
    print("6. Exit")
    
    choice = input("Enter your choice (1-6): ")
    
    sample_df = train_df.sample(n=100000)  
    
    if choice == '1':
        print("\nGenerating word clouds...")
        generate_word_clouds(sample_df)
    elif choice == '2':
        print("\nAnalyzing common phrases...")
        analyze_common_phrases(sample_df)
    elif choice == '3':
        print("\nPlotting distributions...")
        plot_review_length_distribution(sample_df)
        plot_sentiment_distribution(sample_df)
    elif choice == '4':
        print("\nTraining model...")
        model, vectorizer = train_model(train_df)
        print("\nEvaluating on test data...")
        test_df = load_data('test.ft.txt')
        X_test = vectorizer.transform(test_df['review_text'])
        test_pred = model.predict(X_test)
        print(classification_report(test_df['sentiment'], test_pred))
    elif choice == '5':
        print("\nRunning all analyses...")
        generate_word_clouds(sample_df)
        analyze_common_phrases(sample_df)
        plot_review_length_distribution(sample_df)
        plot_sentiment_distribution(sample_df)
        model, vectorizer = train_model(train_df)
        test_df = load_data('test.ft.txt')
        X_test = vectorizer.transform(test_df['review_text'])
        test_pred = model.predict(X_test)
        print(classification_report(test_df['sentiment'], test_pred))
    else:
        print("Exiting...")
        return

if __name__ == "__main__":
    main() 