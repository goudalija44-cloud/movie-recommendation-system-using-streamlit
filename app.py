
import streamlit as st
import pandas as pd
import joblib
import random
from sklearn.metrics.pairwise import cosine_similarity



# Page configuration
st.set_page_config(
    page_title="CineMatch | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# Custom UI styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: #0b0d12;
        color: #f5f5f5;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 13, 18, 0.95);
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #211018, #15151e 60%, #10131b);
        border: 1px solid #33232b;
        border-radius: 24px;
        padding: 42px 38px;
        margin-bottom: 30px;
    }

    .eyebrow {
        color: #ff647c;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .hero h1 {
        font-size: clamp(32px, 5vw, 52px);
        line-height: 1.1;
        font-weight: 800;
        margin: 0 0 15px 0;
        color: #ffffff;
    }

    .hero p {
        color: #b8b8c4;
        font-size: 16px;
        line-height: 1.7;
        max-width: 650px;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 700;
        color: #ffffff;
        margin: 8px 0 5px 0;
    }

    .section-subtitle {
        color: #9ca0ad;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .movie-card {
        background: linear-gradient(145deg, #191c25, #12141b);
        border: 1px solid #2b2e39;
        border-radius: 18px;
        padding: 22px;
        min-height: 165px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }

    .movie-card:hover {
        border-color: #ff647c;
    }

    .movie-number {
        color: #ff647c;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }

    .movie-title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
        line-height: 1.4;
        margin-bottom: 14px;
        overflow-wrap: anywhere;
    }

    .genre-label {
        display: inline-block;
        background: #29202a;
        border: 1px solid #45303c;
        color: #ffc1cb;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 3px 4px 3px 0;
    }

    .empty-state {
        text-align: center;
        background: #141720;
        border: 1px dashed #383b48;
        border-radius: 18px;
        padding: 35px 20px;
        color: #aeb1bd;
        margin-top: 15px;
    }

    div.stButton > button {
        background: #ff526e;
        color: white;
        border: 1px solid #ff526e;
        border-radius: 11px;
        min-height: 45px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #e83f5b;
        border-color: #e83f5b;
        color: white;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #171a23;
        border-color: #343744;
        border-radius: 10px;
    }

    .footer {
        text-align: center;
        color: #777d8b;
        font-size: 12px;
        padding-top: 30px;
    }

    div[data-testid="stMetric"] {
        background: #151821;
        border: 1px solid #292d38;
        border-radius: 14px;
        padding: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aeb1bd;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    @media (max-width: 600px) {
        .hero {
            padding: 28px 22px;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)



# Load saved components
@st.cache_resource
def load_components():
    movies = joblib.load("models/movies.pkl")
    tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
    return movies, tfidf_matrix


movies, tfidf_matrix = load_components()



# Build movie title lookup
indices = pd.Series(
    movies.index,
    index=movies["title"].str.lower()
).drop_duplicates()



# Recommendation function - Existing recommendation logic

def recommend_movies(movie_name, n=5):
    movie_name = movie_name.strip().lower()

    if movie_name not in indices:
        return None

    movie_index = indices[movie_name]

    scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Exclude the selected movie
    scores[movie_index] = -1

    # Find the most similar movies
    top_indices = scores.argsort()[::-1][:n]

    recommendations = movies.iloc[top_indices][
        ["title", "genres"]
    ].copy()

    recommendations["similarity_score"] = [
        round(float(scores[i]), 3)
        for i in top_indices
    ]

    return recommendations.reset_index(drop=True)


# -----------------------------
# Hero section
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR PERSONAL MOVIE GUIDE</div>
    <h1>Find your next<br>favourite movie.</h1>
    <p>
        Tell us what you love watching. Discover films with similar
        stories, genres, casts and themes—all in one place.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Dataset summary
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Movies in library", f"{len(movies):,}")

with col2:
    st.metric("Recommendations per search", "5")

with col3:
    st.metric("Discovery mode", "Content-based")


st.write("")



# Movie selection section
st.markdown(
    '<div class="section-heading">What do you feel like watching?</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Search for a movie or let us pick one for you.'
    '</div>',
    unsafe_allow_html=True
)

movie_titles = sorted(movies["title"].dropna().unique().tolist())

search_col, random_col = st.columns([4, 1])

with search_col:
    search_query = st.text_input(
        "Search movies",
        placeholder="Type a movie title...",
        label_visibility="collapsed"
    )

with random_col:
    random_clicked = st.button(
        "🎲 Surprise me",
        use_container_width=True
    )


# Filter titles using the search field
if search_query.strip():
    filtered_titles = [
        title for title in movie_titles
        if search_query.strip().lower() in title.lower()
    ]
else:
    filtered_titles = movie_titles


# Keep the selection valid after filtering
if not filtered_titles:
    st.warning("No matching movie titles. Try a different search.")
    selected_movie = None
else:
    if random_clicked:
        st.session_state["movie_picker"] = random.choice(filtered_titles)

    if st.session_state.get("movie_picker") not in filtered_titles:
        st.session_state["movie_picker"] = filtered_titles[0]

    selected_movie = st.selectbox(
        "Choose a movie",
        options=filtered_titles,
        key="movie_picker",
        label_visibility="collapsed"
    )



# Recommendation action
if selected_movie:
    if st.button(
        "✨ Discover Similar Movies",
        type="primary",
        use_container_width=True
    ):
        st.session_state["last_movie"] = selected_movie
        st.session_state["show_recommendations"] = True



# Display recommendation cards
if st.session_state.get("show_recommendations", False):
    selected = st.session_state.get("last_movie", "")

    st.divider()

    st.markdown(
        f'<div class="section-heading">'
        f'Movies inspired by {selected}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Here are five movies selected for your next movie night.'
        '</div>',
        unsafe_allow_html=True
    )

    results = recommend_movies(selected, n=5)

    if results is not None and not results.empty:
        for start in range(0, len(results), 2):
            card_columns = st.columns(2)

            for offset, (_, movie) in enumerate(
                results.iloc[start:start + 2].iterrows()
            ):
                with card_columns[offset]:
                    genres = movie["genres"]

                    if isinstance(genres, list):
                        genre_list = genres
                    elif isinstance(genres, str) and genres:
                        genre_list = [
                            genre.strip()
                            for genre in genres.split(",")
                            if genre.strip()
                        ]
                    else:
                        genre_list = []

                    genre_html = "".join(
                        f'<span class="genre-label">{genre}</span>'
                        for genre in genre_list[:5]
                    )

                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="movie-number">
                                PICK {start + offset + 1:02d}
                            </div>
                            <div class="movie-title">
                                {movie["title"]}
                            </div>
                            <div>
                                {genre_html if genre_html else
                                 '<span class="genre-label">Genre unavailable</span>'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.caption(
            "Recommendations are generated from movie content. "
            "Explore the titles and choose what interests you."
        )

        if st.button("🔄 Choose another movie"):
            st.session_state["show_recommendations"] = False
            st.rerun()

    else:
        st.warning("No recommendations were found. Try another movie.")


# Initial empty state
else:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size: 35px;">🍿</div>
        <h3 style="color: #ffffff;">Your next movie night starts here</h3>
        <p>
            Choose a movie above and discover five films
            to add to your watchlist.
        </p>
    </div>
    """, unsafe_allow_html=True)



# Footer
st.markdown("""
<div class="footer">
    CINEMATCH · MOVIE DISCOVERY POWERED BY MACHINE LEARNING
</div>
""", unsafe_allow_html=True)