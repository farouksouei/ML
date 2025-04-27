import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, \
    classification_report
from sklearn.pipeline import Pipeline
import os



class UserEngagementDT:
    def __init__(self, file_path=None, data=None):
        """Initialize with either file path or data."""
        # Create visualization directory
        os.makedirs('charts/engagement_dt', exist_ok=True)

        if file_path:
            self.load_data_from_file(file_path)
        elif data is not None:
            if isinstance(data, pd.DataFrame):
                self.df = data.copy()
            else:
                self.df = pd.DataFrame(data)
        else:
            self.df = pd.DataFrame()

        self.processor = UserEngagementProcessor(data=self.df)

    def load_data_from_file(self, file_path):
        """Load data from CSV file."""
        try:
            self.df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = pd.DataFrame()

    def preprocess_data(self):
        """Preprocess the data using the UserEngagementProcessor."""
        self.processor.clean_data()
        return self

    def run_dt_evaluation(self, cv_folds=5):
        """Run Decision Tree evaluation focusing on criterion (gini vs entropy)."""
        X, y = self.processor.prepare_features()
        feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Compare gini and entropy criterions
        criterions = ['gini', 'entropy']
        results = {}

        for criterion in criterions:
            print(f"\nEvaluating Decision Tree with {criterion} criterion...")

            # Create and train model
            dt = DecisionTreeClassifier(criterion=criterion, random_state=42)
            dt.fit(X_train, y_train)

            # Make predictions
            y_pred = dt.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )

            # Cross-validation
            cv = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
            cv_scores = cross_val_score(dt, X, y, cv=cv, scoring='accuracy')

            # Create classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            # Visualize the decision tree
            plt.figure(figsize=(20, 12))
            plot_tree(dt, feature_names=feature_names,
                      class_names=[str(c) for c in dt.classes_],
                      filled=True, rounded=True, fontsize=10)
            plt.title(f'Decision Tree with {criterion} Criterion', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'charts/engagement_dt/decision_tree_{criterion}.png')
            plt.close()

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=dt.classes_,
                        yticklabels=dt.classes_)
            plt.title(f'Confusion Matrix - Decision Tree ({criterion})', fontsize=14)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'charts/engagement_dt/dt_confusion_matrix_{criterion}.png')
            plt.close()

            # Plot feature importance
            importances = dt.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importances ({criterion})', fontsize=14)
            plt.bar(range(len(importances)),
                    importances[indices],
                    align='center')
            plt.xticks(range(len(importances)),
                       [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'charts/engagement_dt/feature_importances_{criterion}.png')
            plt.close()

            # Plot ROC curve if multi-class
            if len(dt.classes_) > 2:
                self._plot_multiclass_roc(dt, X_test, y_test, criterion)

            # Store results
            results[criterion] = {
                'model': dt,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': dict(zip(feature_names, dt.feature_importances_)),
                'report': report
            }

            # Print results
            print(f"Criterion: {criterion}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Cross-validation mean: {cv_scores.mean():.4f}")
            print(f"Cross-validation std: {cv_scores.std():.4f}")

        # Compare the two criterions
        self._compare_criterions(results)

        # Find optimal tree depth
        self._find_optimal_tree_depth(X, y)

        return results

    def _compare_criterions(self, results):
        """Compare performance of gini vs entropy criterions."""
        # Extract metrics for comparison
        criteria = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']

        # Prepare data for plotting
        comparison_data = []
        for criterion in criteria:
            for metric in metrics:
                comparison_data.append({
                    'Criterion': criterion,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': results[criterion][metric]
                })

        comparison_df = pd.DataFrame(comparison_data)

        # Create bar plot comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Metric', y='Value', hue='Criterion', data=comparison_df)
        plt.title('Decision Tree Performance: Gini vs Entropy', fontsize=16)
        plt.xlabel('Metric', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('charts/engagement_dt/gini_vs_entropy_comparison.png')
        plt.close()

        # Create radar chart
        self._plot_radar_chart(results)

        # Create table with results
        comparison_table = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Mean'],
            'Gini': [results['gini']['accuracy'],
                     results['gini']['precision'],
                     results['gini']['recall'],
                     results['gini']['f1_score'],
                     results['gini']['cv_mean']],
            'Entropy': [results['entropy']['accuracy'],
                        results['entropy']['precision'],
                        results['entropy']['recall'],
                        results['entropy']['f1_score'],
                        results['entropy']['cv_mean']]
        })

        comparison_table.to_csv('charts/engagement_dt/criterion_comparison.csv', index=False)
        print("\nCriterion Comparison:")
        print(comparison_table)

        # Compare feature importance between the two
        self._compare_feature_importance(results)

    def _plot_radar_chart(self, results):
        """Create radar chart comparing the two criterions."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']
        criteria = list(results.keys())

        # Set up the figure
        plt.figure(figsize=(10, 8))

        # Number of metrics we're plotting
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Add axes and adjust layout
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Add metric labels
        plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in metrics], fontsize=12)

        # Plot each criterion
        for criterion in criteria:
            values = [results[criterion][m] for m in metrics]
            values += values[:1]  # Close the circle

            ax.plot(angles, values, linewidth=2, linestyle='solid', label=criterion)
            ax.fill(angles, values, alpha=0.1)

        plt.title('Performance Comparison: Gini vs Entropy', size=16)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('charts/engagement_dt/criterion_radar_chart.png')
        plt.close()

    def _compare_feature_importance(self, results):
        """Compare feature importance between gini and entropy."""
        gini_importance = results['gini']['feature_importance']
        entropy_importance = results['entropy']['feature_importance']

        # Create dataframe for comparison
        features = list(gini_importance.keys())
        importance_df = pd.DataFrame({
            'Feature': features,
            'Gini': [gini_importance[f] for f in features],
            'Entropy': [entropy_importance[f] for f in features]
        })

        # Sort by average importance
        importance_df['Average'] = (importance_df['Gini'] + importance_df['Entropy']) / 2
        importance_df = importance_df.sort_values('Average', ascending=False)

        # Select top 10 features
        top_features = importance_df.head(10)

        # Create comparison plot
        plt.figure(figsize=(14, 8))

        x = np.arange(len(top_features))
        width = 0.35

        plt.bar(x - width / 2, top_features['Gini'], width, label='Gini')
        plt.bar(x + width / 2, top_features['Entropy'], width, label='Entropy')

        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.title('Top 10 Feature Importance: Gini vs Entropy', fontsize=16)
        plt.xticks(x, top_features['Feature'], rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig('charts/engagement_dt/feature_importance_comparison.png')
        plt.close()

    def _find_optimal_tree_depth(self, X, y):
        """Find the optimal tree depth by evaluating different max_depth values."""
        print("\nFinding optimal tree depth...")

        # Try different depths
        depths = range(1, 21)

        # Store accuracy for gini and entropy
        gini_scores = []
        entropy_scores = []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        for depth in depths:
            # Train with gini
            dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=42)
            dt_gini.fit(X_train, y_train)
            y_pred_gini = dt_gini.predict(X_test)
            gini_scores.append(accuracy_score(y_test, y_pred_gini))

            # Train with entropy
            dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
            dt_entropy.fit(X_train, y_train)
            y_pred_entropy = dt_entropy.predict(X_test)
            entropy_scores.append(accuracy_score(y_test, y_pred_entropy))

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(depths, gini_scores, marker='o', linestyle='-', color='blue', label='Gini')
        plt.plot(depths, entropy_scores, marker='s', linestyle='-', color='red', label='Entropy')
        plt.title('Decision Tree Performance vs Max Depth', fontsize=16)
        plt.xlabel('Max Depth', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(depths)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('charts/engagement_dt/optimal_depth.png')
        plt.close()

        # Find optimal depths
        best_gini_depth = depths[np.argmax(gini_scores)]
        best_entropy_depth = depths[np.argmax(entropy_scores)]

        print(f"Optimal depth for gini criterion: {best_gini_depth}")
        print(f"Optimal depth for entropy criterion: {best_entropy_depth}")

        # Train final models with optimal depth
        best_gini_model = DecisionTreeClassifier(
            criterion='gini', max_depth=best_gini_depth, random_state=42)
        best_gini_model.fit(X, y)

        best_entropy_model = DecisionTreeClassifier(
            criterion='entropy', max_depth=best_entropy_depth, random_state=42)
        best_entropy_model.fit(X, y)

        # Visualize optimized trees
        feature_names = X.columns.tolist()
        plt.figure(figsize=(20, 12))
        plot_tree(best_gini_model, feature_names=feature_names,
                  class_names=[str(c) for c in best_gini_model.classes_],
                  filled=True, rounded=True, fontsize=10)
        plt.title(f'Optimized Decision Tree (Gini, depth={best_gini_depth})', fontsize=16)
        plt.tight_layout()
        plt.savefig('charts/engagement_dt/optimized_tree_gini.png')
        plt.close()

        plt.figure(figsize=(20, 12))
        plot_tree(best_entropy_model, feature_names=feature_names,
                  class_names=[str(c) for c in best_entropy_model.classes_],
                  filled=True, rounded=True, fontsize=10)
        plt.title(f'Optimized Decision Tree (Entropy, depth={best_entropy_depth})', fontsize=16)
        plt.tight_layout()
        plt.savefig('charts/engagement_dt/optimized_tree_entropy.png')
        plt.close()

    def _plot_multiclass_roc(self, model, X_test, y_test, criterion):
        """Plot ROC curves for multiclass classification."""
        # Get predictions
        y_score = model.predict_proba(X_test)

        # One vs Rest ROC
        plt.figure(figsize=(10, 8))

        # For each class
        for i, class_name in enumerate(model.classes_):
            # True/False positives
            y_true_binary = (y_test == class_name).astype(int)
            y_score_binary = y_score[:, i]

            fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2,
                     label=f'Class {class_name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Multiclass ROC Curve ({criterion})', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'charts/engagement_dt/multiclass_roc_{criterion}.png')
        plt.close()


if __name__ == "__main__":
    # Set correct file path
    file_path = '../data/edited_skill_exchange_dataset.csv'

    # Create processor and load data
    dt_processor = UserEngagementDT(file_path=file_path)
    dt_processor.preprocess_data()

    # Run the decision tree evaluation focusing on gini vs entropy
    results = dt_processor.run_dt_evaluation(cv_folds=5)

    # Print the final recommendation
    best_criterion = 'gini' if results['gini']['f1_score'] > results['entropy']['f1_score'] else 'entropy'
    best_score = max(results['gini']['f1_score'], results['entropy']['f1_score'])

    print("\nFinal recommendation:")
    print(f"The best criterion for this dataset is '{best_criterion}' with an F1 score of {best_score:.4f}")