import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure chart directories exist
os.makedirs('charts/decision_tree', exist_ok=True)

# Load data
data = pd.read_csv('../data/edited_skill_exchange_dataset.csv')
df = pd.DataFrame(data)

# Data Preprocessing
df['joinedDate'] = pd.to_datetime(df['joinedDate'])
df['days_since_joining'] = (pd.Timestamp.now() - df['joinedDate']).dt.days

# Function to safely split text fields and count items
def count_items(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    items = [item.strip() for item in text.split(',') if item.strip()]
    return len(items)

# Count-based features
df['num_joined_courses'] = df['joinedCourses'].apply(count_items)
df['num_skills'] = df['skills'].apply(count_items)
df['num_desired_skills'] = df['desired_skills'].apply(count_items)

# Convert string lists to actual lists for intersection operations
def string_to_list(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    return [item.strip() for item in text.split(',') if item.strip()]

df['joined_courses_list'] = df['joinedCourses'].apply(string_to_list)
df['skills_list'] = df['skills'].apply(string_to_list)
df['desired_skills_list'] = df['desired_skills'].apply(string_to_list)

# Count overlaps between different skill categories
df['skills_courses_overlap'] = df.apply(
    lambda x: len(set(x['skills_list']).intersection(set(x['joined_courses_list']))), axis=1
)
df['desired_courses_overlap'] = df.apply(
    lambda x: len(set(x['desired_skills_list']).intersection(set(x['joined_courses_list']))), axis=1
)
df['skills_desired_overlap'] = df.apply(
    lambda x: len(set(x['skills_list']).intersection(set(x['desired_skills_list']))), axis=1
)

# Calculate ratios to represent learning progress
df['course_effectiveness'] = df.apply(
    lambda x: x['desired_courses_overlap'] / x['num_joined_courses'] if x['num_joined_courses'] > 0 else 0,
    axis=1
)
df['skills_acquisition_rate'] = df.apply(
    lambda x: x['skills_desired_overlap'] / x['num_desired_skills'] if x['num_desired_skills'] > 0 else 0,
    axis=1
)
df['learning_gap'] = df.apply(
    lambda x: x['num_desired_skills'] - x['skills_desired_overlap'], axis=1
)

# Convert isVerified to numeric
df['isVerified'] = df['isVerified'].map(lambda x: 1 if str(x).lower() == 'true' else 0)

# Features for the model
features = [
    'days_since_joining',
    'num_joined_courses',
    'num_skills',
    'num_desired_skills',
    'skills_courses_overlap',
    'desired_courses_overlap',
    'skills_desired_overlap',
    'course_effectiveness',
    'skills_acquisition_rate',
    'learning_gap',
    'isVerified'
]

# Select features
X = df[features].values
y = df['engagement_level']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# Hyperparameter tuning
param_grid = {
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted'
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train model with best parameters
tree_model = DecisionTreeClassifier(random_state=42, **best_params)
tree_model.fit(X_train, y_train)

# Evaluate model
y_pred = tree_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"RMSE: {rmse:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.tight_layout()
plt.savefig('charts/decision_tree/confusion_matrix.png')
plt.close()

# Feature importance
feature_importance = tree_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.savefig('charts/decision_tree/feature_importance.png')
plt.close()

# Visualize decision tree
plt.figure(figsize=(24, 12))
plot_tree(tree_model, feature_names=features, class_names=list(le.classes_),
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Visualization')
plt.tight_layout()
plt.savefig('charts/decision_tree/tree_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# For deeper trees, we may want a horizontal layout
plt.figure(figsize=(24, 16))
plot_tree(tree_model, feature_names=features, class_names=list(le.classes_),
          filled=True, rounded=True, fontsize=8, orientation='horizontal')
plt.title('Decision Tree Visualization (Horizontal)')
plt.tight_layout()
plt.savefig('charts/decision_tree/tree_visualization_horizontal.png', dpi=300, bbox_inches='tight')
plt.close()

# Text representation of the tree
tree_text = export_text(tree_model, feature_names=features)
with open('charts/decision_tree/tree_text.txt', 'w') as f:
    f.write(tree_text)

# Create a depth vs accuracy analysis
max_depths = range(1, 20)
train_accuracy = []
test_accuracy = []

for depth in max_depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))
    test_accuracy.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(12, 6))
plt.plot(max_depths, train_accuracy, 'o-', label='Training Accuracy')
plt.plot(max_depths, test_accuracy, 'o-', label='Testing Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Effect of Tree Depth on Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('charts/decision_tree/depth_vs_accuracy.png')
plt.close()

# Create a model summary report
with open('charts/decision_tree/model_summary.txt', 'w') as f:
    f.write("Decision Tree Model for User Engagement Prediction\n")
    f.write("==========================================\n\n")

    f.write("1. Data Preprocessing Steps:\n")
    f.write("   - Converted joinedDate to datetime\n")
    f.write("   - Created days_since_joining feature\n")
    f.write("   - Counted items in joinedCourses, skills, and desired_skills\n")
    f.write("   - Calculated overlaps between different skill categories\n")
    f.write("   - Added derived features for learning progress and effectiveness\n\n")

    f.write("2. Features Used:\n")
    for feature in features:
        f.write(f"   - {feature}\n")
    f.write("\n")

    f.write("3. Best Model Parameters:\n")
    for param, value in best_params.items():
        f.write(f"   - {param}: {value}\n")
    f.write("\n")

    f.write("4. Evaluation Metrics:\n")
    f.write(f"   - Accuracy: {accuracy:.4f}\n")
    f.write(f"   - Precision: {precision:.4f}\n")
    f.write(f"   - Recall: {recall:.4f}\n")
    f.write(f"   - F1 Score: {f1:.4f}\n")
    f.write(f"   - RMSE: {rmse:.4f}\n\n")

    f.write("5. Top 5 Most Important Features:\n")
    top_features = importance_df.head(5)
    for idx, row in top_features.iterrows():
        f.write(f"   - {row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n")

print("Decision Tree analysis complete! Files saved in charts/decision_tree/")