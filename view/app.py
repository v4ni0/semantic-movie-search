import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="Movie Recommender", page_icon="🍿", layout="centered")

st.title("🍿 Movie Recommendation System")
st.write("Describe the kind of movie you want to watch, and we'll find the best matches for you!")
description = st.text_area("What are you in the mood for?", placeholder="E.g., Space travel and aliens trying to save humanity...")
top_k = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations"):
    if not description.strip():
        st.warning("Please enter a description first.")
    else:
        with st.spinner("Searching for movies..."):
            try:
                payload = {
                    "description": description,
                    "top_k": top_k
                }
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    
                    if recommendations:
                        st.success("Here are your recommendations!")
                        df = pd.DataFrame(recommendations)
                        if 'score' in df.columns:
                            df['score'] = df['score'].round(3)
                        st.table(df[['title', 'score']].rename(columns={"title": "Movie Title", "score": "Match Score"}))
                    else:
                        st.info("No recommendations found.")
                else:
                    st.error(f"Error from API: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Make sure your FastAPI server is running!")