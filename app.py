from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

app = Flask(__name__)

# Load dataset
df = pd.read_csv('songs.csv')  # title, album, year, image, artist, url, duration

# Combine features for similarity
df['combined'] = df['title'].astype(str) + ' ' + df['artist'].astype(str) + ' ' + df['album'].astype(str)

# TF-IDF vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(input_title, top_n=5):
    # Use fuzzy matching to find closest song title
    all_titles = df['title'].tolist()
    match = process.extractOne(input_title, all_titles, scorer=fuzz.token_sort_ratio)
    
    if not match or match[1] < 60:  # threshold 60%
        return [{"title":"Song not found","artist":"-","album":"-","image":"","url":"#"}]

    matched_title = match[0]
    idx = df[df['title'] == matched_title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [i for i in sim_scores if i[0] != idx][:top_n]

    recommendations = []
    for i in sim_scores:
        song = df.iloc[i[0]]
        recommendations.append({
            "title": song['title'],
            "artist": song['artist'],
            "album": song['album'],
            "image": song['image'],
            "url": song['url']
        })
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form.get('title','')
    recs = get_recommendations(title)
    return jsonify(recs)

if __name__ == '__main__':
    app.run(debug=True)