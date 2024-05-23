from flask import Flask, request, redirect, url_for, render_template, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from bson.objectid import ObjectId

app = Flask(__name__)
app.secret_key = 'secret_key'

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/film_recommendation"
mongo = PyMongo(app)

# Load datasets
df_rating = pd.read_csv("./dataset/Netflix_Dataset_Rating.csv")
df_movies = pd.read_csv("./dataset/Enriched_Movie_data.csv")

# Prepare data
movies_data = df_movies.set_index('Movie_ID').to_dict(orient='index')
movie_ids = list(movies_data.keys())
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

n_users = df_rating['User_ID'].nunique()
n_movies = len(movie_ids)

# Ensure user and movie IDs are within the correct range
max_user_id = df_rating['User_ID'].max()
max_movie_id = df_rating['Movie_ID'].max()

# Adjust n_users and n_movies to fit the actual data range
n_users = max(n_users, max_user_id + 1)
n_movies = max(n_movies, max_movie_id + 1)

# Define MatrixFactorization model
class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_movies, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.movie_factors = torch.nn.Embedding(n_movies, n_factors)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.movie_factors.weight.data.uniform_(0, 0.05)

    def forward(self, user, movie):
        return (self.user_factors(user) * self.movie_factors(movie)).sum(1)

    def predict(self, user, movie):
        return self.forward(user, movie)

# Create and load model
model = MatrixFactorization(n_users, n_movies, n_factors=20)

# Prepare training data
user_ids = torch.tensor(df_rating['User_ID'].values)
movie_indices = torch.tensor([movie_id_to_index[movie_id] for movie_id in df_rating['Movie_ID'].values])
ratings = torch.tensor(df_rating['Rating'].values, dtype=torch.float32)

# Define loss and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
n_epochs = 15
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(user_ids, movie_indices)
    loss = loss_fn(predictions, ratings)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

# Compute movie factors and similarity matrix after training
model.eval()
trained_movie_factors = model.movie_factors.weight.data.cpu().numpy()
similarity_matrix = cosine_similarity(trained_movie_factors)

@app.route('/')
def home():
    user_id = session.get('user_id')
    username = None
    if user_id:
        user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        username = user_data['username'] if user_data else None

    if user_id:
        user_track_data = mongo.db.user_track.find_one({'user_id': user_id})
        if user_track_data and 'movie_ids' in user_track_data:
            saved_movie_ids = user_track_data['movie_ids']
            recommended_movies = get_similar_movies(saved_movie_ids)
        else:
            recommended_movies = random.sample(movie_ids, min(len(movie_ids), 15))
    else:
        recommended_movies = random.sample(movie_ids, min(len(movie_ids), 15))
    
    movies = [{'Movie_ID': movie_id, **movies_data[movie_id]} for movie_id in recommended_movies if movies_data[movie_id]['Poster_Path']]
    return render_template('home.html', movies=movies, username=username)

def get_similar_movies(saved_movie_ids, top_n=15):
    similar_movies = set()
    for movie_id in saved_movie_ids:
        movie_idx = movie_id_to_index[movie_id]
        similar_indices = np.argsort(similarity_matrix[movie_idx])[::-1][1:top_n+1]
        similar_movies.update([movie_ids[idx] for idx in similar_indices])
    similar_movies.difference_update(saved_movie_ids)  # Remove already saved movies
    return list(similar_movies)[:top_n]


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = mongo.db.users.find_one({'username': username})
        if existing_user is None:
            hashpass = generate_password_hash(password, method='pbkdf2:sha256')
            mongo.db.users.insert_one({'username': username, 'password': hashpass})
            return redirect(url_for('login'))
        return 'User already exists!'
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        login_user = mongo.db.users.find_one({'username': username})
        if login_user and check_password_hash(login_user['password'], password):
            session['user_id'] = str(login_user['_id'])
            return redirect(url_for('home'))
        return 'Invalid username/password combination'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    user_id = session.get('user_id')
    username = None
    if user_id:
        user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        username = user_data['username'] if user_data else None

    if movie_id not in movies_data:
        return "Movie ID not found", 404

    movie_data = movies_data[movie_id]
    movie_idx = movie_id_to_index[movie_id]
    similar_indices = np.argsort(similarity_matrix[movie_idx])[::-1][1:11]  # Top 10 similar movies
    similar_movies = [{'Movie_ID': movie_ids[idx], **movies_data[movie_ids[idx]]} for idx in similar_indices if movies_data[movie_ids[idx]]['Poster_Path']]
    return render_template('movie_detail.html', movie_data=movie_data, similar_movies=similar_movies, username=username)

@app.route('/save-movie', methods=['POST'])
def save_movie():
    movie_id = request.form.get('movie_id')
    user_id = session.get('user_id')
    if user_id and movie_id:
        movie_id = int(movie_id)
        mongo.db.user_track.update_one({'user_id': user_id}, {'$addToSet': {'movie_ids': movie_id}}, upsert=True)
        return redirect(url_for('saved_movies_for_user'))
    return 'Unauthorized', 401

@app.route('/saved-movies-for-user')
def saved_movies_for_user():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
    username = user_data['username'] if user_data else None

    user_track_data = mongo.db.user_track.find_one({'user_id': user_id})
    if user_track_data and 'movie_ids' in user_track_data:
        saved_movie_ids = user_track_data['movie_ids']
        movies = [{'Movie_ID': movie_id, **movies_data[movie_id]} for movie_id in saved_movie_ids if movie_id in movies_data]
    else:
        movies = []

    return render_template('saved-movies.html', movies=movies, username=username)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
import base64


@app.route('/matrix_plot')
def matrix_plot():
    # Kullanıcı vektörlerini ve film vektörlerini al
    user_factors = model.user_factors.weight.data.cpu().numpy()
    movie_factors = model.movie_factors.weight.data.cpu().numpy()

    # PCA dönüşümü
    pca = PCA(n_components=2)
    user_pca = pca.fit_transform(user_factors)
    movie_pca = pca.fit_transform(movie_factors)

    # Scatter plotu oluştur
    plt.figure(figsize=(8, 6))
    plt.scatter(user_pca[:, 0], user_pca[:, 1], label='Users')
    plt.scatter(movie_pca[:, 0], movie_pca[:, 1], label='Movies', marker='x')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('User and Movie Embeddings (PCA)')
    plt.legend()

    # Grafiği bayt nesnesine kaydet
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Bayt nesnesini base64 kodlayarak HTML'e yerleştirilebilir hale getir
    img_str = base64.b64encode(buffer.read()).decode()

    # HTML olarak görüntü etiketi oluştur
    img_tag = f'<img src="data:image/png;base64,{img_str}" />'

    return img_tag


if __name__ == '__main__':
    app.run(debug=True)
