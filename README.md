# Text-Similarity-Search-using-TF-IDF-and-Nearest-Neighbors

# Text Similarity Search using TF-IDF and Nearest Neighbors

This Jupyter Notebook demonstrates various approaches for finding similar text entries within a dataset loaded from a CSV file. The core idea is to transform text data into a numerical representation that can be used to calculate the similarity between different text samples and then use this representation to perform efficient searches for similar content.

## Concepts and Methods Explained

The notebook explores several techniques for achieving text similarity search:

### 1. Data Loading and Preprocessing

*   **Loading Data:** The process begins by loading data from a CSV file (`/content/Sample data.csv`) into a pandas DataFrame. Pandas is a powerful library in Python for data manipulation and analysis.
*   **Preprocessing:** To ensure consistent and meaningful comparisons, the text data undergoes preprocessing:
    *   **Lowercase Conversion:** All text is converted to lowercase to treat words like "Urbanic" and "urbanic" as the same.
    *   **Punctuation Removal:** Punctuation marks (like commas, periods, etc.) are removed as they generally don't contribute to the meaning of words and can interfere with similarity calculations.
*   **Concatenating Text Columns:** To create a single representation for each row, the text from all columns is concatenated into a new column called `all_text`. This allows us to consider the entire content of a row when searching for similarity.

### 2. TF-IDF Vectorization

*   **TF-IDF (Term Frequency-Inverse Document Frequency):** This is a widely used technique for converting text into a numerical vector representation.
    *   **Term Frequency (TF):** Measures how often a term appears in a document (in this case, a row in the DataFrame).
    *   **Inverse Document Frequency (IDF):** Measures how unique a term is across all documents. Terms that appear in many documents have a lower IDF, while terms that appear in fewer documents have a higher IDF.
    *   **TF-IDF Score:** The TF-IDF score for a term in a document is the product of its TF and IDF. This weighting scheme gives more importance to terms that are frequent within a document but rare across the entire dataset, thus capturing the distinctiveness of a document's content.
*   **`TfidfVectorizer`:** The `sklearn.feature_extraction.text.TfidfVectorizer` class in scikit-learn is used to perform TF-IDF vectorization. It handles tokenization (splitting text into words), calculating TF and IDF, and generating the TF-IDF matrix.
*   **Parameters:** The `TfidfVectorizer` is used with parameters like `min_df` (minimum document frequency for a term to be considered), `max_df` (maximum document frequency), and `ngram_range` (to include combinations of words, or n-grams, in the vocabulary). These parameters help to filter out very common or very rare words and capture multi-word phrases, which can improve the quality of the vector representation.

### 3. Similarity Calculation (Cosine Similarity)

*   **Cosine Similarity:** This metric is used to measure the similarity between two non-zero vectors in an inner product space. It calculates the cosine of the angle between the two vectors. A cosine similarity close to 1 indicates that the vectors are very similar (pointing in nearly the same direction), while a value close to 0 indicates they are orthogonal (no similarity).
*   **`cosine_similarity`:** The `sklearn.metrics.pairwise.cosine_similarity` function is used to calculate the cosine similarity between the vector of an input word/phrase and the TF-IDF vectors of all the rows in the dataset.

### 4. Finding Similar Rows

*   After calculating the cosine similarities, the indices of the rows with the highest similarity scores are retrieved. These rows are considered the most similar to the input query.

### 5. Optimization with Dimensionality Reduction and Approximate Nearest Neighbors

For larger datasets, performing cosine similarity calculations against every document can be computationally expensive. The notebook explores techniques to make the search more efficient:

*   **Truncated SVD (Singular Value Decomposition):** This is a dimensionality reduction technique. It reduces the number of features (dimensions) in the TF-IDF matrix while preserving as much of the original information as possible. This makes subsequent calculations faster.
*   **Approximate Nearest Neighbors (ANN):** Instead of finding the exact nearest neighbors, ANN algorithms aim to find neighbors that are *approximately* closest. This is much faster for large datasets. The `sklearn.neighbors.NearestNeighbors` class is used to build an ANN model.
*   **Combining SVD and ANN:** By applying SVD to the TF-IDF matrix and then using an ANN model on the reduced-dimensional space, we can significantly speed up the search for similar items.

### 6. Keyword Search (Basic Method)

*   A simple method is included to search for a specific keyword within the dataset. It iterates through each cell of the DataFrame and checks if the keyword (case-insensitive) is present in the string representation of the cell's content. This is a straightforward but less sophisticated approach compared to the vector-based methods.

### 7. Spell Correction using BERT (Example)

*   **BERT (Bidirectional Encoder Representations from Transformers):** A pre-trained language model from the `transformers` library is used to demonstrate how spell correction could be integrated into the search process.
*   **Masked Language Model:** BERT is used in a "masked language model" configuration, where it tries to predict a masked word in a sentence. By providing a keyword and a masked token, BERT can suggest likely correct spellings for the keyword.
*   **Integration with Search:** The corrected keyword from BERT can then be used with the keyword search method to find rows containing the likely correct spelling.

## How to Use

1.  **Upload Data:** Make sure you have a CSV file named `Sample data.csv` in the `/content/` directory of your Colab environment. You can upload your own data following this format.
2.  **Run Cells:** Execute the code cells in the notebook sequentially.
3.  **Modify Inputs:** You can change the `input_word` in the `search_similar_words()` function calls to search for different terms. You can also modify the `keyword` in the basic keyword search section.
4.  **Experiment with Parameters:** Feel free to experiment with the parameters of `TfidfVectorizer`, `TruncatedSVD`, and `NearestNeighbors` to see how they affect the search results and performance.

This notebook provides a solid foundation for building text similarity search capabilities. You can extend this by incorporating more advanced techniques like word embeddings (e.g., Word2Vec, GloVe) or more sophisticated ANN algorithms for even better performance on large datasets.
