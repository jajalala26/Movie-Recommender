import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# Load user features and ratings data
user_info = pd.read_csv('/Users/jaredlee/Desktop/cs135/projectB/data_movie_lens_100k/user_info.csv')
ratings = pd.read_csv('/Users/jaredlee/Desktop/cs135/projectB/data_movie_lens_100k/ratings_all_development_set.csv') 

# Initialize the Surprise Reader and load the dataset
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5), skip_lines=1)
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)


trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train
svd_model = SVD(n_factors=50)
svd_model.fit(trainset)

# Predictions
svd_predictions = svd_model.test(testset)
svd_mae = accuracy.mae(svd_predictions)
print(f"SVD Test MAE: {svd_mae}")

ratings = pd.merge(ratings, user_info[['user_id', 'age', 'is_male']], on='user_id', how='left')

ratings['is_male'] = ratings['is_male'].astype(int)
 
def content_based_adjustment(user_id, age, gender):
    """
    A simple function that adjusts the SVD prediction based on user features.
    The adjustment is a weighted sum of age and gender (simple example).
    You can replace this with a more sophisticated model if needed.
    """
    age_weight = 0.0 
    gender_weight = 0.0 # Test

    adjustment = age_weight * (age - 30) + gender_weight * (gender - 0.5) 
    return adjustment

hybrid_predictions = []
for prediction in svd_predictions:
    uid = prediction.uid
    iid = prediction.iid
    true_r = prediction.r_ui
    svd_pred = prediction.est 
    
    user_data = user_info[user_info['user_id'] == uid]
    age = user_data['age'].values[0]
    gender = user_data['is_male'].values[0]
    
    content_adjustment = content_based_adjustment(uid, age, gender)

    hybrid_pred = svd_pred + content_adjustment
    hybrid_predictions.append((uid, iid, hybrid_pred))
  
# Get MAE
hybrid_mae = np.mean([
    abs(prediction.r_ui - hybrid_pred)  # Compare true rating with hybrid prediction
    for prediction, (_, _, hybrid_pred) in zip(svd_predictions, hybrid_predictions)
])

print(f"Hybrid Model Test MAE: {hybrid_mae}")


