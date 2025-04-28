import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure chart directories exist
os.makedirs('charts/polynomial', exist_ok=True)

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

# Create polynomial features pipeline with logistic regression
polynomial_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomial', PolynomialFeatures()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'))
])

# Hyperparameter tuning
param_grid = {
    'polynomial__degree': [1, 2, 3],
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__solver': ['lbfgs', 'saga']
}

grid_search = GridSearchCV(
    polynomial_pipeline,
    param_grid,
    cv=5,
    scoring='f1_weighted'
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train model with best parameters
best_model = grid_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)

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
plt.title('Polynomial Regression Confusion Matrix')
plt.tight_layout()
plt.savefig('charts/polynomial/confusion_matrix.png')
plt.close()

# Get coefficients from the logistic regression model
poly = best_model.named_steps['polynomial']
clf = best_model.named_steps['classifier']
feature_names = poly.get_feature_names_out(features)
coefficients = clf.coef_

# For each class, show the top features
plt.figure(figsize=(15, 10))
for i, class_name in enumerate(le.classes_):
    coefs = coefficients[i]
    top_coef_indices = np.argsort(np.abs(coefs))[-10:]  # Top 10 features by magnitude
    plt.subplot(len(le.classes_), 1, i+1)
    plt.barh(np.array(feature_names)[top_coef_indices], coefs[top_coef_indices])
    plt.title(f'Top Features for Class: {class_name}')
    plt.tight_layout()
plt.savefig('charts/polynomial/coefficients_by_class.png')
plt.close()

# Create a learning curve
train_sizes = np.linspace(0.1, 0.99, 5)
train_scores = []
test_scores = []

for size in train_sizes:
    X_sub_train, _, y_sub_train, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    # Create a new model with the best parameters
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('polynomial', PolynomialFeatures(degree=best_params['polynomial__degree'])),
        ('classifier', LogisticRegression(
            C=best_params['classifier__C'],
            solver=best_params['classifier__solver'],
            max_iter=1000,
            random_state=42,
            multi_class='multinomial'
        ))
    ])
    model.fit(X_sub_train, y_sub_train)
    train_scores.append(accuracy_score(y_sub_train, model.predict(X_sub_train)))
    test_scores.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
plt.plot(train_sizes, test_scores, 'o-', label='Test Score')
plt.xlabel('Training Set Size Fraction')
plt.ylabel('Accuracy Score')
plt.title('Polynomial Regression Learning Curve')
plt.legend()
plt.grid(True)
plt.savefig('charts/polynomial/learning_curve.png')
plt.close()

# Compare different polynomial degrees
degrees = [1, 2, 3]
degree_scores = []

for degree in degrees:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('polynomial', PolynomialFeatures(degree=degree)),
        ('classifier', LogisticRegression(
            C=best_params['classifier__C'],
            solver=best_params['classifier__solver'],
            max_iter=1000,
            random_state=42,
            multi_class='multinomial'
        ))
    ])
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    degree_scores.append((degree, train_acc, test_acc))

plt.figure(figsize=(10, 6))
plt.plot([d[0] for d in degree_scores], [d[1] for d in degree_scores], 'o-', label='Training Accuracy')
plt.plot([d[0] for d in degree_scores], [d[2] for d in degree_scores], 'o-', label='Testing Accuracy')
plt.xlabel('Polynomial Degree')
plt.ylabel('Accuracy')
plt.title('Effect of Polynomial Degree on Accuracy')
plt.xticks(degrees)
plt.legend()
plt.grid(True)
plt.savefig('charts/polynomial/degree_vs_accuracy.png')
plt.close()

# Create a model summary report
with open('charts/polynomial/model_summary.txt', 'w') as f:
    f.write("Polynomial Regression Model for User Engagement Prediction\n")
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

    f.write("5. Summary of Polynomial Features:\n")
    f.write(f"   - Total number of features after polynomial transformation: {len(feature_names)}\n")
    f.write(f"   - Original features: {len(features)}\n")
    f.write(f"   - Polynomial degree used: {best_params['polynomial__degree']}\n\n")

    f.write("6. Model Interpretation:\n")
    f.write("   - The polynomial regression approach allows for modeling non-linear relationships\n")
    f.write("   - Higher degree polynomials can capture more complex patterns, but may overfit\n")
    f.write("   - The best degree found through cross-validation balances complexity and generalization\n")

print("Polynomial Regression analysis complete! Files saved in charts/polynomial/")