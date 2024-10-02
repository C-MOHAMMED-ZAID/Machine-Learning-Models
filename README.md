# Movie_Recommendation_system

#### Description
This project implements a movie recommendation system using Python. The model recommends movies based on user preferences by analyzing similarities between movie descriptions. The system utilizes the **TF-IDF** vectorization technique and **Cosine Similarity** to provide accurate recommendations.

#### Key Features
- **Data Import**: Loads movie data from a GitHub repository using `pandas.read_csv()`.
- **Feature Extraction**: Utilizes `TfidfVectorizer` to convert movie descriptions into numerical vectors.
- **Similarity Calculation**: Computes cosine similarity between the TF-IDF vectors to find similar movies.
- **User Interaction**: Allows users to input their favorite movie and receive personalized recommendations for the top 10 or 30 similar movies.

#### Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Google Colab** (for development and execution)

#### How to Use
1. Clone the repository or download the notebook.
2. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Open the notebook in Google Colab.
4. Run the code cells, and when prompted, enter your favorite movie title to receive recommendations.

#### Example Code Snippet
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['movie_descriptions'])

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend_movies(movie_title, num_recommendations=10):
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [df['title'][i[0]] for i in sim_scores[1:num_recommendations + 1]]
    return top_movies

# User input for recommendations
user_favorite_movie = input("Enter your favorite movie: ")
recommended_movies = recommend_movies(user_favorite_movie)
print("Recommended Movies:", recommended_movies)
```

#### License
This project is licensed under the MIT License.
