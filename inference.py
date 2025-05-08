# Skill Exchange Platform - Inference Module
# This module loads the trained models and makes predictions for new users

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from datetime import datetime


class SkillExchangeModel:
    def __init__(self, models_dir='models'):
        """Initialize by loading all required model components"""
        self.models_dir = models_dir
        self.kmeans_model = None
        self.scaler = None
        self.knn_models = None
        self.text_vectorizers = None
        self.column_info = None
        # Engagement model components
        self.engagement_model = None
        self.engagement_label_encoder = None
        self.engagement_features = None

        # Load all components
        self.load_components()

    def load_engagement_model(self):
        """Load the engagement prediction XGBoost model and related components"""
        try:
            # Load XGBoost model
            with open(os.path.join(self.models_dir, 'engagement_xgboost_model.pkl'), 'rb') as f:
                self.engagement_model = pickle.load(f)

            # Load label encoder
            with open(os.path.join(self.models_dir, 'engagement_label_encoder.pkl'), 'rb') as f:
                self.engagement_label_encoder = pickle.load(f)

            # Load features list
            with open(os.path.join(self.models_dir, 'engagement_features.pkl'), 'rb') as f:
                self.engagement_features = pickle.load(f)

            print("Engagement XGBoost model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading engagement model: {e}")
            self.engagement_model = None
            self.engagement_label_encoder = None
            self.engagement_features = None
            return False

    def predict_engagement(self, user_data):
        """Predict user engagement level based on skills, courses, and desired skills"""
        if not hasattr(self, 'engagement_model') or self.engagement_model is None:
            success = self.load_engagement_model()
            if not success:
                return {"error": "Engagement model not available"}

        try:
            # Process input data
            features = {}

            # Calculate days since joining
            features['days_since_joining'] = self.calculate_days_since_joined(user_data.get('joinedDate', ''))

            # Process isVerified
            if 'isVerified' in user_data:
                if isinstance(user_data['isVerified'], bool):
                    features['isVerified'] = 1 if user_data['isVerified'] else 0
                elif isinstance(user_data['isVerified'], str):
                    features['isVerified'] = 1 if user_data['isVerified'].lower() == 'true' else 0
                else:
                    features['isVerified'] = int(user_data['isVerified'])
            else:
                features['isVerified'] = 0

            # Count items for skill-related fields
            def count_items(text):
                if not text:
                    return 0
                if isinstance(text, list):
                    return len(text)
                if isinstance(text, str):
                    return len([item.strip() for item in text.split(',') if item.strip()])
                return 0

            def string_to_list(text):
                if not text:
                    return []
                if isinstance(text, list):
                    return text
                if isinstance(text, str):
                    return [item.strip() for item in text.split(',') if item.strip()]
                return []

            # Process counts
            features['num_joined_courses'] = count_items(user_data.get('joinedCourses', ''))
            features['num_skills'] = count_items(user_data.get('skills', ''))
            features['num_desired_skills'] = count_items(user_data.get('desired_skills', ''))

            # Convert to lists for overlap calculations
            joined_courses_list = string_to_list(user_data.get('joinedCourses', ''))
            skills_list = string_to_list(user_data.get('skills', ''))
            desired_skills_list = string_to_list(user_data.get('desired_skills', ''))

            # Calculate overlaps
            features['skills_courses_overlap'] = len(set(skills_list).intersection(set(joined_courses_list)))
            features['desired_courses_overlap'] = len(set(desired_skills_list).intersection(set(joined_courses_list)))
            features['skills_desired_overlap'] = len(set(skills_list).intersection(set(desired_skills_list)))

            # Calculate ratios
            features['course_effectiveness'] = (
                features['desired_courses_overlap'] / features['num_joined_courses']
                if features['num_joined_courses'] > 0 else 0
            )
            features['skills_acquisition_rate'] = (
                features['skills_desired_overlap'] / features['num_desired_skills']
                if features['num_desired_skills'] > 0 else 0
            )
            features['learning_gap'] = features['num_desired_skills'] - features['skills_desired_overlap']

            # Ensure all expected features are present
            X = []
            for feature in self.engagement_features:
                X.append(features.get(feature, 0))

            X = np.array([X])

            # Make prediction
            prediction_encoded = self.engagement_model.predict(X)[0]

            # Convert to probability if possible
            probabilities = {}
            if hasattr(self.engagement_model, 'predict_proba'):
                proba = self.engagement_model.predict_proba(X)[0]
                for i, p in enumerate(proba):
                    label = self.engagement_label_encoder.inverse_transform([i])[0]
                    probabilities[label] = round(float(p), 2)

            # Decode the prediction
            engagement_level = self.engagement_label_encoder.inverse_transform([prediction_encoded])[0]

            result = {"engagement_level": engagement_level, "engagement_probabilities": probabilities,
                      "feature_values": {f: round(float(v), 2) if isinstance(v, (int, float)) else v
                                         for f, v in zip(self.engagement_features, X[0])}}

            # Add the feature values used for transparency

            return result
        except Exception as e:
            import traceback
            return {
                "error": f"Error predicting engagement: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def load_components(self):
        """Load all model components from files"""
        try:
            # Load KMeans model
            with open(os.path.join(self.models_dir, 'kmeans_model.pkl'), 'rb') as f:
                self.kmeans_model = pickle.load(f)
            print("KMeans model loaded successfully")

            # Load scaler
            with open(os.path.join(self.models_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            print("Scaler loaded successfully")

            # Load KNN models if available
            try:
                with open(os.path.join(self.models_dir, 'knn_models.pkl'), 'rb') as f:
                    self.knn_models = pickle.load(f)
                print("KNN models loaded successfully")
            except:
                print("KNN models not available")
                self.knn_models = None

            # Load text vectorizers
            with open(os.path.join(self.models_dir, 'text_vectorizers.pkl'), 'rb') as f:
                self.text_vectorizers = pickle.load(f)
            print("Text vectorizers loaded successfully")

            # Load column information
            with open(os.path.join(self.models_dir, 'column_info.pkl'), 'rb') as f:
                self.column_info = pickle.load(f)
            print("Column information loaded successfully")

            return True

        except Exception as e:
            print(f"Error loading model components: {e}")
            return False

    def calculate_days_since_joined(self, joined_date):
        """Calculate days between joined date and today"""
        if not joined_date:
            return 0

        try:
            joined_date = datetime.strptime(joined_date, '%Y-%m-%d')
            today = datetime.now()
            return (today - joined_date).days
        except:
            return 0

    def prepare_user_features(self, user_data):
        """Process user data to create features for prediction"""
        # Create an empty DataFrame for features
        feature_dfs = []

        # Extract numerical features
        numerical_features = {}

        # Process engagement_level if available
        if 'engagement_level' in user_data:
            if isinstance(user_data['engagement_level'], str):
                # Convert string to numerical value
                engagement_map = {'Low': 0, 'Medium': 1, 'High': 2}
                numerical_features['engagement_level'] = engagement_map.get(user_data['engagement_level'], 1)
            else:
                numerical_features['engagement_level'] = user_data['engagement_level']
        else:
            numerical_features['engagement_level'] = 1  # Default to medium

        # Process isVerified if available
        if 'isVerified' in user_data:
            if isinstance(user_data['isVerified'], bool):
                numerical_features['isVerified'] = 1 if user_data['isVerified'] else 0
            elif isinstance(user_data['isVerified'], str):
                numerical_features['isVerified'] = 1 if user_data['isVerified'].lower() == 'true' else 0
            else:
                numerical_features['isVerified'] = user_data['isVerified']
        else:
            numerical_features['isVerified'] = 0  # Default to not verified

        # Calculate days_since_joined if joinedDate is available
        if 'joinedDate' in user_data:
            numerical_features['days_since_joined'] = self.calculate_days_since_joined(user_data['joinedDate'])
        elif 'days_since_joined' in user_data:
            numerical_features['days_since_joined'] = user_data['days_since_joined']
        else:
            numerical_features['days_since_joined'] = 0  # Default

        numerical_df = pd.DataFrame([numerical_features])
        feature_dfs.append(numerical_df)

        # Process text features
        if 'skills' in user_data and 'skills_tfidf' in self.text_vectorizers:
            skills_text = user_data['skills']
            if isinstance(skills_text, list):
                skills_text = ', '.join(skills_text)
            skills_matrix = self.text_vectorizers['skills_tfidf'].transform([skills_text])
            skills_df = pd.DataFrame(
                skills_matrix.toarray(),
                columns=[f'skill_{i}' for i in range(skills_matrix.shape[1])]
            )
            feature_dfs.append(skills_df)

        if 'joinedCourses' in user_data and 'courses_count' in self.text_vectorizers:
            courses_text = user_data['joinedCourses']
            if isinstance(courses_text, list):
                courses_text = ', '.join(courses_text)
            courses_matrix = self.text_vectorizers['courses_count'].transform([courses_text])
            courses_df = pd.DataFrame(
                courses_matrix.toarray(),
                columns=[f'course_{i}' for i in range(courses_matrix.shape[1])]
            )
            feature_dfs.append(courses_df)

        if 'desired_skills' in user_data and 'desired_tfidf' in self.text_vectorizers:
            desired_text = user_data['desired_skills']
            if isinstance(desired_text, list):
                desired_text = ', '.join(desired_text)
            desired_matrix = self.text_vectorizers['desired_tfidf'].transform([desired_text])
            desired_df = pd.DataFrame(
                desired_matrix.toarray(),
                columns=[f'desired_{i}' for i in range(desired_matrix.shape[1])]
            )
            feature_dfs.append(desired_df)

        # Combine all features
        if feature_dfs:
            all_features_df = pd.concat(feature_dfs, axis=1)

            # Handle expected columns from the scaler
            expected_columns = self.scaler.feature_names_in_

            # Add missing columns with zeros
            for col in expected_columns:
                if col not in all_features_df.columns:
                    all_features_df[col] = 0

            # Keep only expected columns in the correct order
            all_features_df = all_features_df[expected_columns]

            return all_features_df
        else:
            return None

    def predict(self, user_data):
        """Make predictions for a new user"""
        if self.kmeans_model is None:
            return {"error": "Model not loaded properly"}

        # Process the user data
        features_df = self.prepare_user_features(user_data)

        if features_df is None:
            return {"error": "Failed to process user features"}

        # Normalize features
        normalized_features = self.scaler.transform(features_df)

        # Predict cluster
        cluster = int(self.kmeans_model.predict(normalized_features)[0])

        # Calculate distances to all cluster centers for confidence analysis
        distances = self.kmeans_model.transform(normalized_features)[0]
        closest_distance = min(distances)
        confidence_score = 1.0 / (1.0 + closest_distance)  # Convert distance to confidence (0-1)

        # Prepare result
        result = {
            "cluster": cluster,
            "confidence": round(confidence_score, 2),
            "cluster_description": self.get_cluster_description(cluster)
        }

        # If KNN models are available, predict desired skills
        if self.knn_models:
            predicted_skills = []
            skill_scores = {}

            for skill, model in self.knn_models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(normalized_features)[0]
                    skill_scores[skill] = round(proba[1], 2)  # Probability of positive class
                    if proba[1] > 0.5:
                        predicted_skills.append(skill)
                else:
                    prediction = model.predict(normalized_features)[0]
                    skill_scores[skill] = 1.0 if prediction == 1 else 0.0
                    if prediction == 1:
                        predicted_skills.append(skill)

            result["recommended_skills"] = predicted_skills
            result["skill_scores"] = skill_scores

        return result

    def get_cluster_description(self, cluster_id):
        """Return a description for a given cluster based on training data insights"""
        # These descriptions should be based on actual cluster analysis from the training phase
        cluster_descriptions = {
            0: "Technical professionals with high engagement in web technologies",
            1: "Data science enthusiasts with diverse skill interests",
            2: "Software developers focused on programming languages",
            3: "Beginners interested in basic web development",
            4: "Advanced professionals interested in AI and machine learning",
            5: "Business-oriented users seeking technical skills",
            6: "Users looking to expand their technical knowledge"
        }

        return cluster_descriptions.get(cluster_id, "Group of users with similar skill profiles")

    def find_similar_users(self, user_data, top_n=3):
        """Find similar users based on cluster assignment"""
        # This is a simplified implementation - in a real application,
        # you would look up similar users from your database
        if self.kmeans_model is None:
            return []

        features_df = self.prepare_user_features(user_data)

        if features_df is None:
            return []

        normalized_features = self.scaler.transform(features_df)
        cluster = int(self.kmeans_model.predict(normalized_features)[0])

        # In a real implementation, you would query your database for users in the same cluster
        # Here we just return placeholder data
        similar_users = [
            {"user_id": f"user_{cluster}_{i}",
             "match_score": round(0.9 - (i * 0.1), 2),
             "skills": ["Python", "JavaScript", "Data Science"][:3 - i]}
            for i in range(top_n)
        ]

        return similar_users




def main():
    # Example usage
    print("Skill Exchange Platform - Inference Module")
    print("------------------------------------------")

    # Initialize the model
    model = SkillExchangeModel()

    # Example user data based on the provided dataset
    example_user = {
        'user_id': 'new_user',
        'joinedDate': '2023-12-01',
        'joinedCourses': 'Python, Data Science, Machine Learning',
        'skills': 'HTML, CSS, JavaScript, Python',
        'desired_skills': 'Machine Learning, AI, Blockchain',
        'isVerified': True,
        'engagement_level': 'Medium'
    }

    print("\nExample user:")
    for key, value in example_user.items():
        print(f"  {key}: {value}")

    # Make prediction
    result = model.predict(example_user)

    print("\nPrediction result:")
    if "error" in result:
        print(f"  Error: {result['error']}")
    else:
        print(f"  Assigned to Cluster: {result['cluster']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Cluster description: {result['cluster_description']}")

        if "recommended_skills" in result:
            print(f"  Recommended skills to learn: {', '.join(result['recommended_skills'])}")

        # Find similar users
        similar_users = model.find_similar_users(example_user)
        if similar_users:
            print("\nSimilar users:")
            for user in similar_users:
                print(f"  User ID: {user['user_id']}")
                print(f"  Match score: {user['match_score']}")
                print(f"  Skills: {', '.join(user['skills'])}")
                print("")

    # Interactive prediction
    while True:
        print("\nEnter new user information (or type 'exit' to quit):")
        skills = input("Skills (comma-separated): ")
        if skills.lower() == 'exit':
            break

        courses = input("Joined courses (comma-separated): ")
        desired = input("Desired skills (comma-separated): ")

        join_date = input("Join date (YYYY-MM-DD) or leave blank for today: ")

        engagement_str = input("Engagement level (Low/Medium/High): ").capitalize()
        engagement_map = {'Low': 0, 'Medium': 1, 'High': 2}
        engagement = engagement_map.get(engagement_str, 1)

        verified_str = input("Verified (True/False): ").lower()
        verified = verified_str in ('true', 'yes', 'y', '1')

        user_data = {
            'skills': skills,
            'joinedCourses': courses,
            'desired_skills': desired,
            'joinedDate': join_date,
            'engagement_level': engagement,
            'isVerified': verified
        }

        result = model.predict(user_data)

        print("\nPrediction result:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Assigned to Cluster: {result['cluster']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Cluster description: {result['cluster_description']}")

            if "recommended_skills" in result:
                print(f"  Recommended skills to learn: {', '.join(result['recommended_skills'])}")

                if "skill_scores" in result:
                    print("\n  Skill recommendation scores:")
                    for skill, score in sorted(result["skill_scores"].items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"    {skill}: {score}")

            # Find similar users
            similar_users = model.find_similar_users(user_data)
            if similar_users:
                print("\n  Similar users:")
                for user in similar_users:
                    print(f"    User ID: {user['user_id']}")
                    print(f"    Match score: {user['match_score']}")
                    print(f"    Skills: {', '.join(user['skills'])}")
                    print("")
if __name__ == "__main__":
    main()