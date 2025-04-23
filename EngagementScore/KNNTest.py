# Assuming 'processor' is your DataProcessor with cleaned data
from DataProcessor import DataProcessor
from EngagementScore.KNN import SkillKNNRecommender

# Load and clean data
processor = DataProcessor(file_path='../data/edited_skill_exchange_dataset.csv', columns=6)
processor.clean_data()

# Create and train the KNN recommender
recommender = SkillKNNRecommender(processor.df)
recommender.preprocess_data().fit()

# Get recommendations for a user
current_skills = ["Python", "HTML", "CSS"]
desired_skills = ["AI", "Machine Learning"]
recommended_skills = recommender.recommend_skills(current_skills, desired_skills)
print(f"Recommended skills: {recommended_skills}")

# Evaluate model performance
metrics = recommender.evaluate_model(test_indices=range(50))
print(f"Evaluation metrics: {metrics}")

# Visualize recommendations for a specific user
result = recommender.visualize_recommendations(user_id=9800)