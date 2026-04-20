import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

print("Loading mymoviedb.csv...")
# Using the python engine to handle long overviews and complex CSV structure
movies_df = pd.read_csv('mymoviedb.csv', engine='python', on_bad_lines='skip')

# 1. Data Cleaning
movies_df['Overview'] = movies_df['Overview'].fillna('')
movies_df['Genre'] = movies_df['Genre'].fillna('')
# Create a unique Movie Index for internal tracking
movies_df['MovieIndex'] = range(len(movies_df))

# 2. CONTENT-BASED FILTERING (Overview + Genre)
print("Building Content-Based Similarity Engine...")
# We combine Overview and Genre to understand the 'flavor' of the movie
movies_df['Combined_Features'] = movies_df['Genre'] + " " + movies_df['Overview']

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies_df['Combined_Features'])

# To keep memory usage low and the app fast, we'll use a subset for similarity or 
# calculate it on the fly in the app. Here we pre-calculate similarity for the top 3000 movies.
top_movies_indices = movies_df.sort_values('Popularity', ascending=False).head(3000).index
cosine_sim_subset = cosine_similarity(tfidf_matrix[top_movies_indices], tfidf_matrix)

# 3. COLLABORATIVE FILTERING (SVD / Matrix Factorization)
# Since the CSV doesn't have User Ratings, we synthesize them for 500 users
print("Synthesizing Collaborative Filtering Layer (SVD)...")
num_users = 500
np.random.seed(42)

ratings_data = []
for user_id in range(1, num_users + 1):
    # Each user rates 15-40 random popular movies
    num_ratings = np.random.randint(15, 40)
    # We focus ratings on the top 1000 popular movies to make the matrix denser/better
    potential_movies = movies_df.head(1000)['MovieIndex'].values
    rated_movies = np.random.choice(potential_movies, num_ratings, replace=False)
    for m_idx in rated_movies:
        ratings_data.append([user_id, m_idx, np.random.randint(1, 6)])

ratings = pd.DataFrame(ratings_data, columns=['UserID', 'MovieIndex', 'Rating'])

# Build Matrix
pivot_table = ratings.pivot(index='UserID', columns='MovieIndex', values='Rating').fillna(0)
svd = TruncatedSVD(n_components=15, random_state=42)
matrix_fact = svd.fit_transform(pivot_table)
predicted_ratings = np.dot(matrix_fact, svd.components_)
preds_df = pd.DataFrame(predicted_ratings, columns=pivot_table.columns, index=pivot_table.index)

# 4. PREPARE COLD-START DATA (Popularity Engine)
popular_movies = movies_df.sort_values('Popularity', ascending=False).head(50)

# 5. SAVE ARTIFACTS
print("Saving artifacts...")
artifacts = {
    'movies': movies_df,
    'cosine_sim': cosine_sim_subset,
    'sim_indices': top_movies_indices,
    'predictions': preds_df,
    'popular': popular_movies,
    'original_ratings': ratings
}

with open('recommender_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("Success! Training complete.")