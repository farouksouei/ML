import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from datetime import datetime
import os

from EngagementScore.KNN import UserEngagementProcessor


class UserEngagementSVM:
    def __init__(self, file_path=None, data=None):
        """Initialize with either file path or data."""
        # Create visualization directory
        os.makedirs('charts/engagement_svm', exist_ok=True)
        os.makedirs('charts/engagement_svm/kernels', exist_ok=True)

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

    def run_svm_classifier_with_kernels(self, cv_folds=5):
        """Run SVM classifier with different kernels and parameters."""
        X, y = self.processor.prepare_features()

        # Split data consistently for all tests
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Define kernels to test
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        results = {}

        # Test each kernel
        for kernel in kernels:
            print(f"\nTesting SVM with {kernel} kernel...")

            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel=kernel, probability=True))
            ])

            # Train the model
            pipeline.fit(X_train, y_train)

            # Make predictions
            y_pred = pipeline.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )

            # Cross-validation
            cv = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=pipeline.classes_,
                        yticklabels=pipeline.classes_)
            plt.title(f'Confusion Matrix - SVM ({kernel} kernel)', fontsize=14)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'charts/engagement_svm/kernels/svm_{kernel}_confusion_matrix.png')
            plt.close()

            # Store results
            results[kernel] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'trained_model': pipeline
            }

            # Print kernel results
            print(f"Kernel: {kernel}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Cross-validation mean: {cv_scores.mean():.4f}")
            print(f"Cross-validation std: {cv_scores.std():.4f}")

        # Plot comparison of kernel performances
        self._plot_kernel_comparison(results)

        # Find the best kernel based on F1 score
        best_kernel = max(results.items(), key=lambda x: x[1]['f1_score'])

        # Optimize hyperparameters for the best kernel
        optimized_model = self._optimize_hyperparameters(X, y, best_kernel[0])

        return {
            'kernel_results': results,
            'best_kernel': best_kernel[0],
            'optimized_model': optimized_model
        }

    def _plot_kernel_comparison(self, results):
        """Plot a comparison of the different kernel performances."""
        # Extract metrics for comparison
        kernels = list(results.keys())
        accuracy = [results[k]['accuracy'] for k in kernels]
        precision = [results[k]['precision'] for k in kernels]
        recall = [results[k]['recall'] for k in kernels]
        f1 = [results[k]['f1_score'] for k in kernels]
        cv_mean = [results[k]['cv_mean'] for k in kernels]

        # Create a dataframe for easier plotting
        metrics_df = pd.DataFrame({
            'Kernel': np.repeat(kernels, 5),
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Accuracy'] * len(kernels),
            'Value': accuracy + precision + recall + f1 + cv_mean
        })

        # Plot comparison
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Kernel', y='Value', hue='Metric', data=metrics_df)
        plt.title('SVM Performance Comparison Across Kernels', fontsize=16)
        plt.xlabel('Kernel', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.legend(title='Metric', title_fontsize=12, fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('charts/engagement_svm/kernel_comparison.png')
        plt.close()

        # Create a more detailed radar chart for each kernel
        self._plot_radar_chart(results)

        # Plot cross-validation scores comparison
        plt.figure(figsize=(12, 6))
        for kernel in kernels:
            plt.plot(results[kernel]['cv_scores'], marker='o', linestyle='-', label=kernel)

        plt.title('Cross-validation Scores Across Folds by Kernel', fontsize=14)
        plt.xlabel('Fold', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(title='Kernel')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('charts/engagement_svm/cv_scores_comparison.png')
        plt.close()

    def _plot_radar_chart(self, results):
        """Create radar charts for visualizing model performance across metrics."""
        # Extract metrics for the radar chart
        kernels = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']

        # Set up the figure
        plt.figure(figsize=(12, 10))

        # Number of metrics we're plotting
        num_metrics = len(metrics)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Add axes and adjust layout
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Add metric labels
        plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in metrics], fontsize=12)

        # Plot each kernel
        for kernel in kernels:
            # Extract values for this kernel
            values = [results[kernel][m] for m in metrics]
            values += values[:1]  # Close the circle

            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=kernel)
            ax.fill(angles, values, alpha=0.1)

        plt.title('SVM Performance Metrics by Kernel', size=16)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('charts/engagement_svm/radar_chart_comparison.png')
        plt.close()

    def _optimize_hyperparameters(self, X, y, best_kernel):
        """Optimize hyperparameters for the best kernel."""
        print(f"\nOptimizing hyperparameters for {best_kernel} kernel...")

        # Create parameter grid based on the kernel
        if best_kernel == 'linear':
            param_grid = {
                'svm__C': [0.1, 1, 10, 100],
                'svm__class_weight': ['balanced', None]
            }
        elif best_kernel == 'poly':
            param_grid = {
                'svm__C': [0.1, 1, 10],
                'svm__degree': [2, 3, 4],
                'svm__gamma': ['scale', 'auto', 0.1, 1],
                'svm__class_weight': ['balanced', None]
            }
        else:  # 'rbf' or 'sigmoid'
            param_grid = {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.1, 1],
                'svm__class_weight': ['balanced', None]
            }

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=best_kernel, probability=True))
        ])

        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1_weighted', verbose=1
        )
        grid_search.fit(X, y)

        # Print best parameters
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")

        # Plot parameter comparison
        self._plot_grid_search_results(grid_search)

        # Evaluate best model
        best_model = grid_search.best_estimator_
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=best_model.classes_,
                    yticklabels=best_model.classes_)
        plt.title(f'Confusion Matrix - SVM (Optimized {best_kernel})', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'charts/engagement_svm/optimized_svm_confusion_matrix.png')
        plt.close()

        # Print optimized results
        print(f"Optimized {best_kernel} SVM:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def _plot_grid_search_results(self, grid_search):
        """Plot grid search results to visualize hyperparameter performance."""
        # Extract results
        results = pd.DataFrame(grid_search.cv_results_)

        # Clean parameter names for plotting
        param_cols = [col for col in results.columns if col.startswith('param_')]
        for col in param_cols:
            clean_name = col.replace('param_svm__', '')
            results[clean_name] = results[col].astype(str)

        # Create a summary for the most important parameters
        important_params = ['C', 'gamma', 'degree', 'class_weight']
        present_params = [p for p in important_params if f'param_svm__{p}' in results.columns]

        if len(present_params) > 0:
            for param in present_params:
                param_col = f'param_svm__{param}'
                if param_col in results.columns:
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x=param, y='mean_test_score', data=results)
                    plt.title(f'Impact of {param} on Model Performance', fontsize=14)
                    plt.xlabel(param, fontsize=12)
                    plt.ylabel('F1 Score (weighted)', fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(f'charts/engagement_svm/param_{param}_impact.png')
                    plt.close()

        # Plot comparing top 5 combinations
        top_combinations = results.sort_values('mean_test_score', ascending=False).head(5)

        plt.figure(figsize=(14, 6))
        param_combinations = []
        for i, row in top_combinations.iterrows():
            combo = []
            for param in present_params:
                param_col = f'param_svm__{param}'
                if param_col in row:
                    combo.append(f"{param}={row[param_col]}")
            param_combinations.append("\n".join(combo))

        sns.barplot(x=param_combinations, y=top_combinations['mean_test_score'])
        plt.title('Top 5 Hyperparameter Combinations', fontsize=14)
        plt.xlabel('Parameter Combination', fontsize=12)
        plt.ylabel('F1 Score (weighted)', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('charts/engagement_svm/top_parameter_combinations.png')
        plt.close()

    def compare_with_knn(self, knn_results):
        """Compare SVM performance with KNN."""
        # Run SVM with different kernels
        svm_results = self.run_svm_classifier_with_kernels()

        # Get best kernel results
        best_kernel = svm_results['best_kernel']
        svm_metrics = svm_results['optimized_model']

        # Prepare comparison data
        models = ['KNN', f'SVM ({best_kernel})']
        accuracy = [knn_results['accuracy'], svm_metrics['accuracy']]
        precision = [knn_results['precision'], svm_metrics['precision']]
        recall = [knn_results['recall'], svm_metrics['recall']]
        f1 = [knn_results['f1_score'], svm_metrics['f1_score']]

        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': np.repeat(models, 4),
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'] * 2,
            'Value': accuracy + precision + recall + f1
        })

        # Plot comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Model', y='Value', hue='Metric', data=comparison_df)
        plt.title('Model Performance Comparison: KNN vs SVM', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.legend(title='Metric', title_fontsize=12, fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('charts/engagement_svm/knn_vs_svm_comparison.png')
        plt.close()

        # Create a table of results for easier comparison
        comparison_table = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'KNN': [knn_results['accuracy'], knn_results['precision'],
                   knn_results['recall'], knn_results['f1_score']],
            f'SVM ({best_kernel})': [svm_metrics['accuracy'], svm_metrics['precision'],
                                    svm_metrics['recall'], svm_metrics['f1_score']]
        })

        # Save comparison table
        comparison_table.to_csv('charts/engagement_svm/model_comparison.csv', index=False)

        print("\nModel Comparison:")
        print(comparison_table)

        return {
            'comparison_table': comparison_table,
            'svm_results': svm_results,
            'knn_results': knn_results
        }


# Example usage code
if __name__ == "__main__":
    # Load data
    file_path = '../data/edited_skill_exchange_dataset.csv'

    # Run KNN first
    knn_processor = UserEngagementProcessor(file_path=file_path)
    knn_processor.clean_data()
    knn_results = knn_processor.run_knn_classifier(n_neighbors=6, cv_folds=5)

    # Print KNN results
    print("\nClassification Results for KNN:")
    print(f"Accuracy: {knn_results['accuracy']:.4f}")
    print(f"Precision: {knn_results['precision']:.4f}")
    print(f"Recall: {knn_results['recall']:.4f}")
    print(f"F1 Score: {knn_results['f1_score']:.4f}")
    print(f"Cross-validation mean accuracy: {knn_results['cv_mean']:.4f}")
    print(f"Cross-validation std: {knn_results['cv_std']:.4f}")

    # Run SVM
    svm_processor = UserEngagementSVM(file_path=file_path)
    svm_processor.preprocess_data()

    # Compare KNN and SVM
    comparison = svm_processor.compare_with_knn(knn_results)