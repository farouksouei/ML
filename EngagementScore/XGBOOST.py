import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
import os
import shap
from EngagementScore.KNN import UserEngagementProcessor


class UserEngagementXGB:
    def __init__(self, file_path=None, data=None):
        """Initialize with either file path or data."""
        # Create visualization directory
        os.makedirs('charts/engagement_xgb', exist_ok=True)

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

    def run_xgboost_evaluation(self, cv_folds=5):
        """Run XGBoost evaluation with different learning rates."""
        X, y = self.processor.prepare_features()
        feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Compare different learning rates
        learning_rates = [0.01, 0.1, 0.3]
        results = {}

        for lr in learning_rates:
            print(f"\nEvaluating XGBoost with learning rate {lr}...")

            # Create and train model
            xgb_model = xgb.XGBClassifier(
                learning_rate=lr,
                n_estimators=100,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            xgb_model.fit(X_train, y_train)

            # Make predictions
            y_pred = xgb_model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )

            # Cross-validation
            cv = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
            cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')

            # Create classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=np.unique(y),
                        yticklabels=np.unique(y))
            plt.title(f'Confusion Matrix - XGBoost (lr={lr})', fontsize=14)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'charts/engagement_xgb/xgb_confusion_matrix_lr{lr}.png')
            plt.close()

            # Plot feature importance
            plt.figure(figsize=(12, 8))
            xgb.plot_importance(xgb_model, max_num_features=10)
            plt.title(f'Feature Importance (lr={lr})', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'charts/engagement_xgb/feature_importance_lr{lr}.png')
            plt.close()

            # Store results
            results[f'lr_{lr}'] = {
                'model': xgb_model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': dict(zip(feature_names, xgb_model.feature_importances_)),
                'report': report
            }

            # Print results
            print(f"Learning rate: {lr}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Cross-validation mean: {cv_scores.mean():.4f}")
            print(f"Cross-validation std: {cv_scores.std():.4f}")

            # SHAP values for model interpretability
            try:
                explainer = shap.Explainer(xgb_model)
                shap_values = explainer(X_test)

                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                plt.title(f'SHAP Feature Importance (lr={lr})', fontsize=14)
                plt.tight_layout()
                plt.savefig(f'charts/engagement_xgb/shap_importance_lr{lr}.png')
                plt.close()
            except Exception as e:
                print(f"Error generating SHAP values: {e}")

        # Compare the different learning rates
        self._compare_learning_rates(results)

        # Find optimal parameters
        self._find_optimal_parameters(X, y)

        return results

    def _compare_learning_rates(self, results):
        """Compare performance of different learning rates."""
        # Extract metrics for comparison
        learning_rates = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']

        # Prepare data for plotting
        comparison_data = []
        for lr in learning_rates:
            for metric in metrics:
                comparison_data.append({
                    'Learning Rate': lr.replace('lr_', ''),
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': results[lr][metric]
                })

        comparison_df = pd.DataFrame(comparison_data)

        # Create bar plot comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Metric', y='Value', hue='Learning Rate', data=comparison_df)
        plt.title('XGBoost Performance: Learning Rate Comparison', fontsize=16)
        plt.xlabel('Metric', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('charts/engagement_xgb/learning_rate_comparison.png')
        plt.close()

        # Create table with results
        comparison_table = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Mean'],
            '0.01': [results['lr_0.01']['accuracy'],
                     results['lr_0.01']['precision'],
                     results['lr_0.01']['recall'],
                     results['lr_0.01']['f1_score'],
                     results['lr_0.01']['cv_mean']],
            '0.1': [results['lr_0.1']['accuracy'],
                    results['lr_0.1']['precision'],
                    results['lr_0.1']['recall'],
                    results['lr_0.1']['f1_score'],
                    results['lr_0.1']['cv_mean']],
            '0.3': [results['lr_0.3']['accuracy'],
                    results['lr_0.3']['precision'],
                    results['lr_0.3']['recall'],
                    results['lr_0.3']['f1_score'],
                    results['lr_0.3']['cv_mean']]
        })

        comparison_table.to_csv('charts/engagement_xgb/learning_rate_comparison.csv', index=False)
        print("\nLearning Rate Comparison:")
        print(comparison_table)

        # Compare feature importance between different learning rates
        self._compare_feature_importance(results)

    def _compare_feature_importance(self, results):
        """Compare feature importance between different learning rates."""
        # Get feature importance for each learning rate
        importances = {}
        for lr_key, result in results.items():
            importances[lr_key] = result['feature_importance']

        # Create dataframe for top 10 features across all models
        all_features = set()
        for imp in importances.values():
            all_features.update(imp.keys())

        # Get top 10 features by average importance
        avg_importance = {}
        for feature in all_features:
            avg_importance[feature] = np.mean([imp.get(feature, 0) for imp in importances.values()])

        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_feature_names = [f[0] for f in top_features]

        # Create comparison plot
        importance_df = pd.DataFrame(index=top_feature_names)

        for lr_key, imp in importances.items():
            importance_df[lr_key.replace('lr_', '')] = [imp.get(feature, 0) for feature in top_feature_names]

        # Plot
        plt.figure(figsize=(14, 10))
        importance_df.plot(kind='barh', figsize=(14, 10))
        plt.title('Top 10 Feature Importance Across Learning Rates', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.legend(title='Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('charts/engagement_xgb/feature_importance_comparison.png')
        plt.close()

    def _find_optimal_parameters(self, X, y):
        """Find the optimal parameters for XGBoost using GridSearchCV."""
        print("\nFinding optimal XGBoost parameters...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Define parameter grid
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Create base model
        xgb_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )

        # Grid search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=3,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # Get best parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best parameters: {best_params}")
        print(f"Best CV score: {best_score:.4f}")

        # Train model with best parameters
        best_model = xgb.XGBClassifier(**best_params, random_state=42)
        best_model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        print("\nOptimized XGBoost model performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Save best results
        with open('charts/engagement_xgb/best_parameters.txt', 'w') as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best CV score: {best_score:.4f}\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Test Precision: {precision:.4f}\n")
            f.write(f"Test Recall: {recall:.4f}\n")
            f.write(f"Test F1 Score: {f1:.4f}\n")

        # Plot feature importance for best model
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(best_model, max_num_features=10)
        plt.title('Feature Importance (Optimized Model)', fontsize=14)
        plt.tight_layout()
        plt.savefig('charts/engagement_xgb/optimized_feature_importance.png')
        plt.close()

        # Create confusion matrix for best model
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y),
                    yticklabels=np.unique(y))
        plt.title('Confusion Matrix - Optimized XGBoost', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('charts/engagement_xgb/optimized_confusion_matrix.png')
        plt.close()

        return {
            'best_model': best_model,
            'best_params': best_params,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def compare_with_tree(self, tree_results):
        """Compare XGBoost performance with Decision Tree."""
        # Run XGBoost
        xgb_results = self.run_xgboost_evaluation()

        # Get best learning rate
        best_lr_key = max(xgb_results.keys(), key=lambda k: xgb_results[k]['f1_score'])
        xgb_metrics = xgb_results[best_lr_key]

        # Get best tree criterion
        best_criterion = 'gini' if tree_results['gini']['f1_score'] > tree_results['entropy']['f1_score'] else 'entropy'
        tree_metrics = tree_results[best_criterion]

        # Prepare comparison data
        models = [f'Tree ({best_criterion})', f'XGBoost ({best_lr_key.replace("lr_", "")})']
        accuracy = [tree_metrics['accuracy'], xgb_metrics['accuracy']]
        precision = [tree_metrics['precision'], xgb_metrics['precision']]
        recall = [tree_metrics['recall'], xgb_metrics['recall']]
        f1 = [tree_metrics['f1_score'], xgb_metrics['f1_score']]
        cv_mean = [tree_metrics['cv_mean'], xgb_metrics['cv_mean']]

        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': np.repeat(models, 5),
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Mean'] * 2,
            'Value': accuracy + precision + recall + f1 + cv_mean
        })

        # Plot comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Model', y='Value', hue='Metric', data=comparison_df)
        plt.title('Model Performance Comparison: Decision Tree vs XGBoost', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.legend(title='Metric', title_fontsize=12, fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('charts/engagement_xgb/tree_vs_xgboost_comparison.png')
        plt.close()

        # Create a table of results for easier comparison
        comparison_table = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Mean'],
            f'Tree ({best_criterion})': [tree_metrics['accuracy'], tree_metrics['precision'],
                                         tree_metrics['recall'], tree_metrics['f1_score'],
                                         tree_metrics['cv_mean']],
            f'XGBoost ({best_lr_key.replace("lr_", "")})': [xgb_metrics['accuracy'], xgb_metrics['precision'],
                                                            xgb_metrics['recall'], xgb_metrics['f1_score'],
                                                            xgb_metrics['cv_mean']]
        })

        # Save comparison table
        comparison_table.to_csv('charts/engagement_xgb/tree_vs_xgboost_comparison.csv', index=False)

        print("\nModel Comparison:")
        print(comparison_table)

        return {
            'comparison_table': comparison_table,
            'xgb_results': xgb_results,
            'tree_results': tree_results
        }


if __name__ == "__main__":
    # Set correct file path
    file_path = '../data/edited_skill_exchange_dataset.csv'

    # First run Decision Tree
    from EngagementScore.TREE import UserEngagementDT

    dt_processor = UserEngagementDT(file_path=file_path)
    dt_processor.preprocess_data()
    tree_results = dt_processor.run_dt_evaluation(cv_folds=5)

    # Run XGBoost and compare
    xgb_processor = UserEngagementXGB(file_path=file_path)
    xgb_processor.preprocess_data()
    comparison = xgb_processor.compare_with_tree(tree_results)