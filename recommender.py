from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load data
columns = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=columns)

# Load movie information
movie_info_cols = ['movie_id', 'movie_title']
movies_info = pd.read_csv('u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=movie_info_cols)

# Create a new user with the next available user ID
new_user_id = ratings['user_id'].max() + 1

@app.route('/')
def index():
    user_ratings = ratings[ratings['user_id'] == new_user_id].merge(movies_info, how='left', left_on='item_id', right_on='movie_id')
    return render_template('index.html', user_ratings=user_ratings)

@app.route('/rate_movie', methods=['POST'])
def rate_movie():
    global ratings  # Declare ratings as a global variable

    movie_title_input = request.form['movie_title']
    rating = int(request.form['rating'])

    # Make the input case-insensitive and remove the year information
    movie_title_input = movie_title_input.lower().split('(')[0].strip()

    # Find item_id corresponding to the entered movie_title
    movie_titles_lower = movies_info['movie_title'].str.lower()
    match_indices = movie_titles_lower.str.contains(movie_title_input)

    if match_indices.any():
        # Select the first matched movie
        item_id = movies_info.loc[match_indices, 'movie_id'].iloc[0]

        user_ratings_data = {'user_id': [new_user_id], 'item_id': [item_id], 'rating': [rating]}
        user_ratings = pd.DataFrame(user_ratings_data)

        # Append the new user ratings to the existing dataset
        ratings = pd.concat([ratings, user_ratings])

        return redirect(url_for('index'))
    else:
        return render_template('error.html', message=f"Movie '{movie_title_input}' not found in the dataset. Please enter a valid movie title.")

@app.route('/recommendations')
def recommendations():
    # Create a user-item matrix by taking the average rating for duplicate entries
    user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating', aggfunc='mean').fillna(0)

    # Load movie information
    movie_info_cols = ['movie_id', 'movie_title']
    movies_info = pd.read_csv('u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=movie_info_cols)

    # Calculate similarity scores for the new user with all other users
    user_ids = ratings['user_id'].unique()
    user_ids = user_ids[user_ids != new_user_id]

    similarity_scores = []
    for user_id in user_ids:
        both_viewed = both_rated(new_user_id, user_id)
        if len(both_viewed) > 0:
            similarity = pearson_correlation(new_user_id, user_id, both_viewed)
            similarity_scores.append((user_id, similarity))

    # Choose the top N users with highest similarity scores
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    N = 50
    top_similarity_scores = similarity_scores[:N]

    predicted_ratings = calculate_predicted_ratings(user_item_matrix, top_similarity_scores)

    # Create a DataFrame with movie_id, movie_title, and predicted_rating
    recommendations = pd.DataFrame({'movie_id': user_item_matrix.columns,
                                    'movie_title': movies_info['movie_title'],
                                    'predicted_rating': predicted_ratings})

    # Filter movies with predicted ratings > 3.5
    threshold = 3.5
    recommended_movies = recommendations[recommendations['predicted_rating'] > threshold]

    # Sort recommended movies by predicted rating in descending order
    recommended_movies = recommended_movies.sort_values(by='predicted_rating', ascending=False)

    return render_template('recommendations.html', recommended_movies=recommended_movies)

# Define similarity calculation functions

# Find movies that both users have rated
def both_rated(u1, u2):
    movies_u1 = ratings[ratings['user_id'] == u1]['item_id'].values
    movies_u2 = ratings[ratings['user_id'] == u2]['item_id'].values
    return np.intersect1d(movies_u1, movies_u2)

# Calculate average ratings for both users on common movies
def rating_average(u1, u2, both_viewed):
    ratings_u1 = ratings[(ratings['user_id'] == u1) & (ratings['item_id'].isin(both_viewed))]['rating'].values
    ratings_u2 = ratings[(ratings['user_id'] == u2) & (ratings['item_id'].isin(both_viewed))]['rating'].values
    return np.mean(ratings_u1), np.mean(ratings_u2)

# Calculate Pearson correlation coefficient
def pearson_correlation(u1, u2, both_viewed):
    avg_u1, avg_u2 = rating_average(u1, u2, both_viewed)
    ratings_u1 = ratings[(ratings['user_id'] == u1) & (ratings['item_id'].isin(both_viewed))]['rating'].values
    ratings_u2 = ratings[(ratings['user_id'] == u2) & (ratings['item_id'].isin(both_viewed))]['rating'].values
    
    numerator = np.sum((ratings_u1 - avg_u1) * (ratings_u2 - avg_u2))
    denominator_u1 = np.sqrt(np.sum((ratings_u1 - avg_u1)**2))
    denominator_u2 = np.sqrt(np.sum((ratings_u2 - avg_u2)**2))
    
    if denominator_u1 == 0 or denominator_u2 == 0:
        return 0  # Handle division by zero
    else:
        return numerator / (denominator_u1 * denominator_u2)

# Calculate predicted ratings for movies using aggregate function
def calculate_predicted_ratings(user_item_matrix, top_similarity_scores):
    predicted_ratings = []

    for movie_id in user_item_matrix.columns:
        similarity_rating = 0
        sum_similarity = 0

        for user_id, similarity in top_similarity_scores:
            similarity_score = similarity
            user_rating = user_item_matrix.loc[user_id, movie_id]

            if user_rating > 0:
                similarity_rating += similarity_score * user_rating
                sum_similarity += similarity_score

        # Handle division by zero
        if sum_similarity == 0:
            predicted_rating = 0
        else:
            predicted_rating = similarity_rating / sum_similarity
            predicted_rating = round(predicted_rating, 2)

        predicted_ratings.append(predicted_rating)

    return predicted_ratings
if __name__ == '__main__':
    app.run(debug=True)