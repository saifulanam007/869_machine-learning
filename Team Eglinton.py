import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import time
from psutil import virtual_memory
import os

# Check RAM availability
ram_gb = virtual_memory().total / 1e9
print('Your system has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

# Set the file paths (Ensure these files are present in the same directory as this script)
train_data_path = 'MMA869_large_training_FinalPC_v1.csv'
test_data_path = 'MMA869_test_data_without_response_FinalPC_v1.csv'

start_time = time.time()

# Load datasets
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Focus on relevant columns
train_data = train_data[['overall_rating', 'firm', 'date_review', 'job_title', 'location', 'headline', 'pros', 'cons']]
test_data = test_data[['firm', 'date_review', 'job_title', 'location', 'headline', 'pros', 'cons']]

# Data preprocessing
train_data['headline'] = train_data['headline'].fillna('').astype(str)
train_data['pros'] = train_data['pros'].fillna('').astype(str)
train_data['cons'] = train_data['cons'].fillna('').astype(str)

test_data['headline'] = test_data['headline'].fillna('').astype(str)
test_data['pros'] = test_data['pros'].fillna('').astype(str)
test_data['cons'] = test_data['cons'].fillna('').astype(str)

train_data['combined_text'] = train_data['headline'] + ' ' + train_data['pros'] + ' ' + train_data['cons']
test_data['combined_text'] = test_data['headline'] + ' ' + test_data['pros'] + ' ' + test_data['cons']

# Drop rows with invalid date_review values
train_data = train_data[~train_data['date_review'].str.contains('#')]
test_data = test_data[~test_data['date_review'].str.contains('#')]

# Ensure proper date conversion
train_data['date_review'] = pd.to_datetime(train_data['date_review'], errors='coerce')
test_data['date_review'] = pd.to_datetime(test_data['date_review'], errors='coerce')

# Drop rows where date conversion failed
train_data = train_data.dropna(subset=['date_review'])
test_data = test_data.dropna(subset=['date_review'])

# Extract date features
train_data['year'] = train_data['date_review'].dt.year
train_data['month'] = train_data['date_review'].dt.month
train_data['day'] = train_data['date_review'].dt.day
train_data['day_of_week'] = train_data['date_review'].dt.dayofweek
train_data['is_weekend'] = train_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
train_data['quarter'] = train_data['date_review'].dt.quarter
train_data['week_of_year'] = train_data['date_review'].dt.isocalendar().week

test_data['year'] = test_data['date_review'].dt.year
test_data['month'] = test_data['date_review'].dt.month
test_data['day'] = test_data['date_review'].dt.day
test_data['day_of_week'] = test_data['date_review'].dt.dayofweek
test_data['is_weekend'] = test_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
test_data['quarter'] = test_data['date_review'].dt.quarter
test_data['week_of_year'] = test_data['date_review'].dt.isocalendar().week

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def compute_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

train_data['headline_sentiment'] = train_data['headline'].apply(compute_sentiment)
train_data['pros_sentiment'] = train_data['pros'].apply(compute_sentiment)
train_data['cons_sentiment'] = train_data['cons'].apply(compute_sentiment)

test_data['headline_sentiment'] = test_data['headline'].apply(compute_sentiment)
test_data['pros_sentiment'] = test_data['pros'].apply(compute_sentiment)
test_data['cons_sentiment'] = test_data['cons'].apply(compute_sentiment)

# Text length features
train_data['headline_length'] = train_data['headline'].apply(len)
train_data['pros_length'] = train_data['pros'].apply(len)
train_data['cons_length'] = train_data['cons'].apply(len)

test_data['headline_length'] = test_data['headline'].apply(len)
test_data['pros_length'] = test_data['pros'].apply(len)
test_data['cons_length'] = test_data['cons'].apply(len)

# Define feature columns
additional_features = ['headline_sentiment', 'pros_sentiment', 'cons_sentiment', 'headline_length', 'pros_length', 'cons_length', 'day_of_week', 'is_weekend', 'quarter', 'week_of_year']
categorical_features = ['firm', 'location', 'job_title']
text_features = 'combined_text'
date_features = ['year', 'month', 'day']

# Interaction features
train_data['pros_cons_sentiment_diff'] = train_data['pros_sentiment'] - train_data['cons_sentiment']
test_data['pros_cons_sentiment_diff'] = test_data['pros_sentiment'] - test_data['cons_sentiment']

train_data['pros_headline_sentiment_diff'] = train_data['pros_sentiment'] - train_data['headline_sentiment']
test_data['pros_headline_sentiment_diff'] = test_data['pros_sentiment'] - test_data['headline_sentiment']

additional_features.append('pros_cons_sentiment_diff')
additional_features.append('pros_headline_sentiment_diff')

# Tokenize and pad text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data['combined_text'])
train_sequences = tokenizer.texts_to_sequences(train_data['combined_text'])
test_sequences = tokenizer.texts_to_sequences(test_data['combined_text'])

max_sequence_length = 500
X_train_text = pad_sequences(train_sequences, maxlen=max_sequence_length)
X_test_text = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Train Word2Vec model
w2v_model = Word2Vec(sentences=[text.split() for text in train_data['combined_text']], vector_size=100, window=5, min_count=1, workers=4)
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features),
        ('num', StandardScaler(), date_features + additional_features)
    ])

# Define the training set
X_meta = train_data.drop(columns=['overall_rating', 'date_review', 'headline', 'pros', 'cons'])
y = train_data['overall_rating']

# Cross-validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)
val_mse_scores = []
val_r2_scores = []

# Move data to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CombinedDataset(Dataset):
    def __init__(self, meta_data, text_data, labels=None):
        self.meta_data = meta_data
        self.text_data = torch.tensor(text_data, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return self.meta_data.shape[0]

    def __getitem__(self, idx):
        meta_data = torch.tensor(self.meta_data[idx].toarray(), dtype=torch.float32).squeeze()
        if self.labels is not None:
            return meta_data, self.text_data[idx], self.labels[idx]
        else:
            return meta_data, self.text_data[idx]

# Define a simple neural network in PyTorch
class SimpleNN(nn.Module):
    def __init__(self, meta_input_dim, embedding_matrix, text_input_dim):
        super(SimpleNN, self).__init__()
        self.meta_fc1 = nn.Linear(meta_input_dim, 128)
        self.meta_fc2 = nn.Linear(128, 64)
        
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        self.text_lstm = nn.LSTM(100, 100, batch_first=True, bidirectional=True)
        self.text_fc = nn.Linear(100 * 2, 64)

        self.fc1 = nn.Linear(64 + 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, meta_data, text_data):
        meta_out = torch.relu(self.meta_fc1(meta_data))
        meta_out = torch.relu(self.meta_fc2(meta_out))
        
        text_out = self.embedding(text_data)
        text_out, _ = self.text_lstm(text_out)
        text_out = torch.max(text_out, dim=1)[0]
        text_out = torch.relu(self.text_fc(text_out))

        combined_out = torch.cat((meta_out, text_out), dim=1)
        combined_out = torch.relu(self.fc1(combined_out))
        combined_out = torch.relu(self.fc2(combined_out))
        combined_out = self.fc3(combined_out)
        return combined_out

# KFold Cross-validation training
for train_index, val_index in kf.split(X_meta):
    X_meta_train, X_meta_val = X_meta.iloc[train_index], X_meta.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Split the text sequences for cross-validation
    X_train_text_cv, X_val_text_cv = X_train_text[train_index], X_train_text[val_index]

    # Preprocess the data
    X_meta_train_transformed = preprocessor.fit_transform(X_meta_train)
    X_meta_val_transformed = preprocessor.transform(X_meta_val)

    # Create PyTorch datasets and dataloaders
    train_dataset = CombinedDataset(X_meta_train_transformed, X_train_text_cv, y_train.values)
    val_dataset = CombinedDataset(X_meta_val_transformed, X_val_text_cv, y_val.values)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize the network and move it to the GPU
    model = SimpleNN(X_meta_train_transformed.shape[1], embedding_matrix, max_sequence_length).to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for meta_data, text_data, labels in train_loader:
            meta_data, text_data, labels = meta_data.to(device), text_data.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(meta_data, text_data)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Validation
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for meta_data, text_data, labels in val_loader:
            meta_data, text_data, labels = meta_data.to(device), text_data.to(device), labels.to(device)
            outputs = model(meta_data, text_data)
            val_predictions.append(outputs.squeeze().cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    val_predictions = np.concatenate(val_predictions)
    val_labels = np.concatenate(val_labels)
    mse_val = mean_squared_error(val_labels, val_predictions)
    r2_val = r2_score(val_labels, val_predictions)
    
    val_mse_scores.append(mse_val)
    val_r2_scores.append(r2_val)

print("Validation MSE scores:", val_mse_scores)
print("Mean Validation MSE:", np.mean(val_mse_scores))
print("Validation R2 scores:", val_r2_scores)
print("Mean Validation R2:", np.mean(val_r2_scores))

# Train final model on all data
X_meta_train_transformed = preprocessor.fit_transform(X_meta)
X_test_transformed = preprocessor.transform(test_data.drop(columns=['date_review', 'headline', 'pros', 'cons']))

final_train_dataset = CombinedDataset(X_meta_train_transformed, X_train_text, y.values)
final_test_dataset = CombinedDataset(X_test_transformed, X_test_text)

final_train_loader = DataLoader(final_train_dataset, batch_size=16, shuffle=True)

model = SimpleNN(X_meta_train_transformed.shape[1], embedding_matrix, max_sequence_length).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the final model
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for meta_data, text_data, labels in final_train_loader:
        meta_data, text_data, labels = meta_data.to(device), text_data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(meta_data, text_data)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(final_train_loader):.4f}')

# Predict on the test set
final_test_loader = DataLoader(final_test_dataset, batch_size=16, shuffle=False)

model.eval()
test_predictions = []
with torch.no_grad():
    for meta_data, text_data in final_test_loader:
        meta_data, text_data = meta_data.to(device), text_data.to(device)
        outputs = model(meta_data, text_data)
        test_predictions.append(outputs.squeeze().cpu().numpy())

test_predictions = np.concatenate(test_predictions)
test_predictions = np.clip(test_predictions, 1, 5)  # Ensure predictions are within the range [1, 5]

# Save predictions to CSV
# Ensure 'ID_num' exists in test data, if not create it
if 'ID_num' not in test_data.columns:
    test_data['ID_num'] = range(1, len(test_data) + 1)

test_data['prediction'] = test_predictions
output_path = 'predictions.csv'
test_data[['ID_num', 'prediction']].to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")

# Verify the file is in the current directory
print("Current directory files:", os.listdir())




#Q2 

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# Download stopwords from NLTK
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

# Load the dataset
large_train_path = "/content/MMA869_large_training_FinalPC_v1.csv"
large_train = pd.read_csv(large_train_path)

# Combine text columns
large_train['combined_text'] = large_train['headline'].fillna('') + ' ' + large_train['pros'].fillna('') + ' ' + large_train['cons'].fillna('')

# Function to calculate phrase frequency
def calculate_phrase_frequency(texts, ngram_range=(2, 2)):
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    phrase_counts = X.toarray().sum(axis=0)
    phrases = vectorizer.get_feature_names_out()
    return Counter(dict(zip(phrases, phrase_counts)))

# Create word clouds and calculate word frequencies for each star level
star_levels = large_train['overall_rating'].unique()
phrase_frequencies = {}

for star in sorted(star_levels):
    star_reviews = large_train[large_train['overall_rating'] == star]['combined_text']
    phrase_frequencies[star] = calculate_phrase_frequency(star_reviews)

# Plot phrase frequencies for each star level
for star, freq in phrase_frequencies.items():
    top_phrases = freq.most_common(20)
    phrases, counts = zip(*top_phrases)

    plt.figure(figsize=(10, 6))
    plt.bar(phrases, counts, color='skyblue')
    plt.xlabel('Phrases')
    plt.ylabel('Frequency')
    plt.title(f'{int(star)}-Star Reviews: Top 20 Phrases')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
train_data_path = 'MMA869_large_training_FinalPC_v1.csv'
train_data = pd.read_csv(train_data_path)

# Preprocess the data
train_data['combined_text'] = train_data['headline'].fillna('') + ' ' + train_data['pros'].fillna('') + ' ' + train_data['cons'].fillna('')

# Generate word clouds for each star level
star_levels = [1, 2, 3, 4, 5]
wordclouds = {}

for star in star_levels:
    text = ' '.join(train_data[train_data['overall_rating'] == star]['combined_text'])
    wordclouds[star] = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot word clouds
fig, axs = plt.subplots(1, 5, figsize=(20, 10))
for i, star in enumerate(star_levels):
    axs[i].imshow(wordclouds[star], interpolation='bilinear')
    axs[i].axis('off')
    axs[i].set_title(f'{star}-Star Reviews')

plt.tight_layout()
plt.show()



#Q3



import pandas as pd

# Load data
smalltrain_path = "/content/MMA869_small_training_FinalPC_v1.csv"
largetrain_path = "/content/MMA869_large_training_FinalPC_v1.csv"

small_data = pd.read_csv(smalltrain_path)
large_data = pd.read_csv(largetrain_path)

# Concatenate the small and large training datasets
train_data = pd.concat([small_data, large_data], ignore_index=True)


# 1. Preprocess

# Combine the text from 'headline' and 'cons' columns
train_data['combined_text'] = train_data['headline'].fillna('') + ' ' + train_data['cons'].fillna('')

# Verify the combined text
train_data['combined_text'].head()


# 2. Extract for smaller sample

# Extract a smaller sample for low-rated companies
low_rated_sample = train_data[train_data['overall_rating'].isin([1, 2])].sample(5000, random_state=42)
medium_rated_sample = train_data[train_data['overall_rating'] == 3].sample(5000, random_state=42)
high_rated_sample = train_data[train_data['overall_rating'].isin([4, 5])].sample(5000, random_state=42)

# Extract the text data for the sample
low_rated_text_sample = low_rated_sample['combined_text']
medium_rated_text_sample = medium_rated_sample['combined_text']
high_rated_text_sample = high_rated_sample['combined_text']


# 3. Apply TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_significant_phrases(text_data, n=10, ngram_range=(2, 3)):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()

    # Sum up the TF-IDF scores for each phrase
    tfidf_scores = dense.sum(axis=0).tolist()[0]
    phrase_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
    phrase_scores = sorted(phrase_scores, key=lambda x: x[1], reverse=True)
    return phrase_scores[:n]

# Extract significant phrases for each rating group
significant_phrases_low_rated = extract_significant_phrases(low_rated_text_sample)
significant_phrases_medium_rated = extract_significant_phrases(medium_rated_text_sample)
significant_phrases_high_rated = extract_significant_phrases(high_rated_text_sample)

# Display results
significant_phrases_low_rated, significant_phrases_medium_rated, significant_phrases_high_rated

import matplotlib.pyplot as plt

def plot_significant_phrases(significant_phrases, title):
    phrases, scores = zip(*significant_phrases)
    plt.figure(figsize=(10, 6))
    plt.barh(phrases, scores)
    plt.xlabel('TF-IDF Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

# Plot the significant phrases for each rating group
plot_significant_phrases(significant_phrases_low_rated, 'Significant Complaints in Low-Rated Companies (1-2)')
plot_significant_phrases(significant_phrases_medium_rated, 'Significant Complaints in Medium-Rated Companies (3)')
plot_significant_phrases(significant_phrases_high_rated, 'Significant Complaints in High-Rated Companies (4-5)')


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load dataset
train_data_path = 'MMA869_large_training_FinalPC_v1.csv'
train_data = pd.read_csv(train_data_path)

# Preprocess the data
train_data['combined_text'] = train_data['headline'].fillna('') + ' ' + train_data['pros'].fillna('') + ' ' + train_data['cons'].fillna('')
train_data = train_data.dropna(subset=['combined_text', 'overall_rating'])

# Vectorize the text data
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(train_data['combined_text'])
y = train_data['overall_rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and analyze
train_data['predicted_rating'] = model.predict(X)

# Group by predicted rating
bins = [1, 2, 3, 4, 5, 6]
labels = ['1', '2', '3', '4', '5']
train_data['rating_group'] = pd.cut(train_data['predicted_rating'], bins=bins, labels=labels, right=False)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Compute sentiment scores for each group
sentiments = []
for group in labels:
    group_data = train_data[train_data['rating_group'] == group]['combined_text']
    sentiment = [analyzer.polarity_scores(text)['compound'] for text in group_data]
    sentiments.append(np.mean(sentiment))

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(labels, sentiments, color='skyblue')
plt.xlabel('Rating Group')
plt.ylabel('Average Sentiment Score')
plt.title('Average Sentiment Score by Predicted Rating Group')
plt.show()

# Display insights in a table
insights = pd.DataFrame({
    'Rating Group': labels,
    'Average Sentiment Score': sentiments
})
print(insights)

# Discuss causality
causality_discussion = """
It is important to note that while our analysis reveals a correlation between sentiment scores and overall ratings,
it does not necessarily imply a causal relationship. The sentiments expressed in the reviews might be influenced
by various external factors, and the ratings might be subjective based on individual experiences. Therefore,
we cannot conclude that improving certain aspects (e.g., adding restrooms or offering sushi) will directly
result in higher ratings without further controlled experiments to establish causality.
"""
print(causality_discussion)
