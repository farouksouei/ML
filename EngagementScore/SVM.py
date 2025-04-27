import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs('charts/svm', exist_ok=True)

print("Loading data...")
# Load data
df = pd.read_csv('../data/edited_skill_exchange_dataset.csv')

# Basic preprocessing
print("Preprocessing data...")
df['joinedDate'] = pd.to_datetime(df['joinedDate'])
df['days_since_joining'] = (pd.Timestamp.now() - df['joinedDate']).dt.days

# Convert text fields to counts
def count_items(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len([item.strip() for item in text.split(',') if item.strip()])

# Simplified feature set - only use counts and basic features
df['num_joined_courses'] = df['joinedCourses'].apply(count_items)
df['num_skills'] = df['skills'].apply(count_items)
df['num_desired_skills'] = df['desired_skills'].apply(count_items)
df['isVerified'] = df['isVerified'].map(lambda x: 1 if str(x).lower() == 'true' else 0)

# Simplify features to reduce complexity
features = [
    'days_since_joining',
    'num_joined_courses',
    'num_skills',
    'num_desired_skills',
    'isVerified'
]

print("Preparing data for modeling...")
# Prepare features and target
X = df[features]
y = df['engagement_level']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data with smaller test size for faster execution
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Training model with simplified parameters...")
# Create a simpler pipeline with default parameters
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear', C=1.0, probability=True))  # Use linear kernel for faster training
])

# Train the model directly without grid search
pipeline.fit(X_train, y_train)

print("Evaluating model...")
# Make predictions
y_pred = pipeline.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Create confusion matrix
print("Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVM')
plt.tight_layout()
plt.savefig('charts/svm/confusion_matrix.png')
plt.close()

# Simple feature importance calculation
print("Calculating basic feature importance...")
feature_importance = []

for i, feature in enumerate(features):
    # Calculate feature importance using coefficients (works for linear SVM)
    if hasattr(pipeline.named_steps['svm'], 'coef_'):
        importance = np.abs(pipeline.named_steps['svm'].coef_).mean(axis=0)
    else:
        # For non-linear kernels, use a simple approach
        importance = np.ones(len(features))  # Placeholder
    feature_importance = importance

# Create feature importance plot
if isinstance(feature_importance, np.ndarray) and len(feature_importance) == len(features):
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance for SVM')
    plt.tight_layout()
    plt.savefig('charts/svm/feature_importance.png')
    plt.close()

# Create summary report
print("Creating summary report...")
with open('charts/svm/model_summary.txt', 'w') as f:
    f.write("SVM Model for User Engagement Prediction (Simplified Version)\n")
    f.write("=====================================================\n\n")

    f.write("1. Data Preprocessing Steps:\n")
    f.write("   - Converted joinedDate to datetime\n")
    f.write("   - Created days_since_joining feature\n")
    f.write("   - Counted items in joinedCourses, skills, and desired_skills\n\n")

    f.write("2. Features Used:\n")
    for feature in features:
        f.write(f"   - {feature}\n")
    f.write("\n")

    f.write("3. Model Parameters:\n")
    f.write("   - kernel: linear\n")
    f.write("   - C: 1.0\n\n")

    f.write("4. Evaluation Metrics:\n")
    f.write(f"   - Accuracy: {accuracy:.4f}\n")
    f.write(f"   - Precision: {precision:.4f}\n")
    f.write(f"   - Recall: {recall:.4f}\n")
    f.write(f"   - F1 Score: {f1:.4f}\n\n")

print("Done!")