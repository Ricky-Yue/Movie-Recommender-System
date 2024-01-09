# Movie-Recommender-System
Si Yuan (Ricky) Yue

**Introduction**
This project involves the implementation of a movie recommender system using a
user-based collaborative filtering algorithm. The primary goal is to recommend movies
to a user based on the preferences of other users who exhibit high similarity in movie
ratings. The project utilizes a Python script named recommender.py along with two data
files (u.data and u.item) sourced from the MovieLens 100k dataset. The dataset
comprises 100,000 ratings (1-5) from 943 users on 1682 movies.

**Project Components**
Python File: recommender.py
The recommender.py script serves as the core of the movie recommender system. It
involves the following key components:
1. User-based Collaborative Filtering Algorithm: The algorithm identifies users
with similar movie preferences to the current user and recommends movies
based on their ratings.
2. Top N Users with Highest Similarity Scores: The system selects the top N
(N=50) users with the highest similarity scores to the current user. This step is
crucial for enhancing the accuracy of movie recommendations.
3. Threshold-based Recommendation: Movies with predicted ratings above a
threshold of 3.5 are recommended to the user. This threshold ensures that only
movies likely to be well-received by the user are suggested.
4. Descending Order of Ratings: The recommended movies are listed in
descending order of predicted ratings, allowing the user to explore the
highest-rated suggestions first.
5. Data Files:
                   ● u.data: This file contains user ratings for movies, including information such as
                     user ID, movie ID, rating, and timestamp.
                   ● u.item: The movie information file provides details like movie ID and movie title.
   
**Implementation Details**
User Input and Ratings
The system prompts the user to rate 10 movies on a scale of 1 to 5. The ratings are
stored in a DataFrame, and the new user's preferences are incorporated into the
existing dataset.
Similarity Calculation
The algorithm calculates Pearson correlation coefficients to determine the similarity
between the new user and all other users. The top N users with the highest similarity
scores are selected for further processing.
Predicted Ratings
Predicted ratings for movies are calculated based on the ratings of similar users. The
aggregate function considers the ratings of the selected users and produces predicted
ratings for each movie.
Recommendation Display
The system filters and sorts the recommended movies based on the specified threshold.
The user is presented with a list of movies exceeding the threshold, sorted in
descending order of predicted ratings.

**How to Run the Program**
1. To run the program user must install the following:
a. pip install Flask
2. To run the program first open up the ‘recommender.py’ file in an IDE of your
choice.
3. Next run the debugger or press ‘f5’
4. Open your web browser and go to http://127.0.0.1:5000/. You should see the
Movie Rating App interface where users can rate movies and view
recommendations.
5. Enter 10 movie ratings to view recommended movie(s).
6. Movies with predicted rating above 3.5 based on ‘User-Based Collaborative
Algorithm will be displayed.
