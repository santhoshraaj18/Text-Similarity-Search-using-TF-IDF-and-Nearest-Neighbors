# -*- coding: utf-8 -*-
"""mlmodel test for 1panel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1X-KVZNUIVCqNZ7ayzK29BqT0gOYVBusm
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV Data
df = pd.read_csv("/content/Sample data.csv")

# Preprocess Text Data
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply preprocessing to all columns
for col in df.columns:
    df[col] = df[col].apply(preprocess_text)

# Concatenate all text columns into one
# Concatenate all text columns into one
df['all_text'] = df.astype(str).apply(' '.join, axis=1)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['all_text'])

# Define the search similar words function
def search_similar_words(input_word, top_n=5):
    input_word_vector = tfidf_vectorizer.transform([input_word])
    cosine_similarities = cosine_similarity(input_word_vector, tfidf_matrix).flatten()
    related_indices = cosine_similarities.argsort()[::-1][:top_n]
    return df.iloc[related_indices]

# Example usage
similar_words_df = search_similar_words("Urbanic")
print(similar_words_df)

"""**important code**"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV Data
df = pd.read_csv("/content/Sample data.csv")

# Preprocess Text Data
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply preprocessing to all columns
for col in df.columns:
    df[col] = df[col].apply(preprocess_text)

# Concatenate all text columns into one
df['all_text'] = pd.concat([df[col].astype(str) for col in df.columns], axis=1).apply(lambda row: ' '.join(row), axis=1)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['all_text'])

# Define the search similar words function
def search_similar_words(input_word):
    input_word_vector = tfidf_vectorizer.transform([input_word])
    cosine_similarities = cosine_similarity(input_word_vector, tfidf_matrix).flatten()
    related_indices = cosine_similarities.argsort()[::-1]
    similar_rows = df.iloc[related_indices]
    return similar_rows[cosine_similarities > 0]  # Return all matches

# Example usage
similar_words_df = search_similar_words("male")
print(similar_words_df)

from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# TF-IDF Vectorization with optimized parameters
tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['all_text'])

# Dimensionality reduction with Truncated SVD
svd = TruncatedSVD(n_components=100)
tfidf_matrix_svd = svd.fit_transform(tfidf_matrix)

# Approximate Nearest Neighbors search
ann_model = NearestNeighbors(n_neighbors=100, algorithm='auto')
ann_model.fit(tfidf_matrix_svd)

def search_similar_words(input_word):
    input_word_vector = tfidf_vectorizer.transform([input_word])
    input_word_vector_svd = svd.transform(input_word_vector)
    distances, indices = ann_model.kneighbors(input_word_vector_svd)
    all_similar_rows = df.iloc[indices.flatten()]
    # Filter rows with non-zero cosine similarity
    similar_rows = all_similar_rows[distances.flatten() > 0]
    return similar_rows


# Example usage
similar_words_df = search_similar_words("female")
print(similar_words_df)

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# Load CSV Data
df = pd.read_csv("/content/Sample data.csv")

# Preprocess Text Data
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply preprocessing to all columns
for col in df.columns:
    df[col] = df[col].apply(preprocess_text)

# Concatenate all text columns into one
df['all_text'] = pd.concat([df[col].astype(str) for col in df.columns], axis=1).apply(lambda row: ' '.join(row), axis=1)

# TF-IDF Vectorization with optimized parameters
tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['all_text'])

# Dimensionality reduction with Truncated SVD
svd = TruncatedSVD(n_components=100)
tfidf_matrix_svd = svd.fit_transform(tfidf_matrix)

# Approximate Nearest Neighbors search
ann_model = NearestNeighbors(n_neighbors=len(df), algorithm='auto')
ann_model.fit(tfidf_matrix_svd)

def search_similar_words(input_word):
    input_word_vector = tfidf_vectorizer.transform([input_word])
    input_word_vector_svd = svd.transform(input_word_vector)
    distances, indices = ann_model.kneighbors(input_word_vector_svd)
    all_similar_rows = df.iloc[indices.flatten()]
    # Filter rows with non-zero cosine similarity
    similar_rows = all_similar_rows[distances.flatten() > 0]
    return similar_rows

# Example usage
similar_words_df = search_similar_words("13257387743")
print(similar_words_df)

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load CSV Data
df = pd.read_csv("/content/Sample data.csv")

# Preprocess Text Data
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply preprocessing to all columns
for col in df.columns:
    df[col] = df[col].apply(preprocess_text)

# Concatenate all text columns into one
df['all_text'] = pd.concat([df[col].astype(str) for col in df.columns], axis=1).apply(lambda row: ' '.join(row), axis=1)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['all_text'])

# Approximate Nearest Neighbors search with a different algorithm
ann_model = NearestNeighbors(n_neighbors=len(df), algorithm='auto')
ann_model.fit(tfidf_matrix)

def search_similar_words(input_word):
    input_word_vector = tfidf_vectorizer.transform([input_word])
    distances, indices = ann_model.kneighbors(input_word_vector)
    all_similar_rows = df.iloc[indices.flatten()]
    # Filter rows with non-zero cosine similarity
    similar_rows = all_similar_rows[distances.flatten() > 0]
    return similar_rows

# Example usage
similar_words_df = search_similar_words("zara")
print(similar_words_df)

!pip install pyspellchecker

import pandas as pd

# Load your CSV file
df = pd.read_csv('/content/Sample data.csv')

# Define your keyword
keyword = 'zara'

# Use the apply function to check each cell for the keyword
mask = df.applymap(lambda x: keyword.lower() in str(x).lower())

# Get the rows where the keyword is found
result = df[mask.any(axis=1)]

# Print the result
print(result)

/content/Sample data.csv

!pip install transformers

import pandas as pd
from transformers import BertForMaskedLM, BertTokenizer

# Load your CSV file
df = pd.read_csv('/content/Sample data.csv')

# Define your keyword
keyword = 'frui'

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Prepare the inputs for the model
inputs = tokenizer.encode(f'{keyword} {tokenizer.mask_token}', return_tensors='pt')

# Get the prediction from the model
prediction = model(inputs)[0]

# Get the index of the masked token
masked_index = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

# Get the top 5 predictions
top_5_predictions = prediction[0, masked_index].topk(5).indices.tolist()

# Get the corrected keyword
corrected_keyword = tokenizer.decode(top_5_predictions[0])

# Use the apply function to check each cell for the keyword
mask = df.applymap(lambda x: corrected_keyword.lower() in str(x).lower())

# Get the rows where the keyword is found
result = df[mask.any(axis=1)]

# Print the result
print(result)