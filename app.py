import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Styling with shadow-sm, rounded-xl, and border-gray-200
st.set_page_config(page_title="Global Movie AI", layout="wide")

st.markdown("""
    <style>
    .stButton > button {
        border-radius: 0.75rem !important; /* rounded-xl */
        border: 1px solid #e5e7eb !important; /* border-gray-200 */
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important; /* shadow-sm */
        padding: 0.5rem 1rem;
    }
    .movie-card {
        background-color: white;
        border: 1px solid #e5e7eb; /* border-gray-200 */
        border-radius: 0.75rem; /* rounded-xl */
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
        padding: 10px;
        text-align: center;
        transition: transform 0.2s;
    }
    .movie-card:hover { transform: scale(1.02); }
    img { border-radius: 0.5rem; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    with open('recommender_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

art = load_data()
df = art['movies']
preds = art['predictions']

def show_grid(movie_list, title="Recommended for You"):
    st.subheader(title)
    cols = st.columns(5)
    for i, (_, row) in enumerate(movie_list.iterrows()):
        with cols[i % 5]:
            # Use poster URL from your CSV
            st.markdown(f"""
            <div class="movie-card">
                <img src="{row['Poster_Url']}" width="100%">
                <div style="font-weight:bold; font-size:14px;">{row['Title']}</div>
                <div style="font-size:11px; color:#6b7280;">{row['Genre'].split(',')[0]}</div>
            </div>
            """, unsafe_allow_html=True)

st.title("🎬 Global Movie Recommendation Engine")
mode = st.radio("User Mode:", ["New User (Cold-Start)", "Existing Profile (AI SVD)"], horizontal=True)

if mode == "New User (Cold-Start)":
    st.info("Showing globally trending movies based on popularity scores.")
    # Extract unique genres from your data
    all_genres = sorted(list(set([g.strip() for sub in df['Genre'].str.split(',') for g in sub if g.strip()])))
    selected_genre = st.selectbox("What do you feel like watching?", all_genres)
    
    # Filter by genre and popularity
    results = df[df['Genre'].str.contains(selected_genre)].sort_values('Popularity', ascending=False).head(10)
    show_grid(results, title=f"Top Trending in {selected_genre}")

else:
    user_id = st.selectbox("Select User Profile ID:", preds.index)
    if st.button("Generate Recommendations"):
        # Get SVD predictions for this specific user
        user_preds = preds.loc[user_id].sort_values(ascending=False)
        # Filter out what they've already "seen" (rated in synth data)
        seen = art['original_ratings'][art['original_ratings']['UserID'] == user_id]['MovieIndex'].values
        rec_indices = user_preds[~user_preds.index.isin(seen)].head(10).index
        recs = df.iloc[rec_indices]
        show_grid(recs, title="AI-Driven Picks (Matrix Factorization)")