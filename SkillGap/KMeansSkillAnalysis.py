import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import os
from collections import Counter

class KMeansSkillAnalysis:
    def __init__(self, data_processor):
        """Initialize with a DataProcessor instance containing cleaned data."""
        self.data_processor = data_processor
        self.df = data_processor.df
        self.features = None
        self.scaled_features = None
        self.scaler = StandardScaler()
        self.kmeans_models = {}
        self.labels = {}

        # Create directory for cluster visualizations
        os.makedirs('charts/clusters', exist_ok=True)

    def prepare_features(self):
        """Prepare features for clustering."""
        # Extract numerical features
        feature_cols = [
            'skill_count_current',
            'skill_count_desired',
            'skill_count_target',
            'skill_gap_count',
            'skill_overlap_count',
            'learning_efficiency'
        ]

        # One-hot encode skills
        skill_cols = ['current_skills', 'desired_skills', 'target_skills']
        skill_dummies = []

        for col in skill_cols:
            # Get all unique skills
            all_skills = set()
            for skills in self.df[col]:
                all_skills.update(skills)

            # Create dummy variables for top skills
            top_skills = Counter([skill for skills_list in self.df[col] for skill in skills_list]).most_common(10)
            top_skill_names = [skill[0] for skill in top_skills]

            for skill in top_skill_names:
                self.df[f"{col}_{skill}"] = self.df[col].apply(lambda x: 1 if skill in x else 0)
                feature_cols.append(f"{col}_{skill}")

        self.features = self.df[feature_cols]

        # Handle missing values
        self.features = self.features.fillna(0)

        # Scale features
        self.scaled_features = self.scaler.fit_transform(self.features)

        return self.scaled_features

    def find_optimal_k(self, max_k=15):
        """Find the optimal number of clusters using various metrics."""
        k_values = range(2, max_k + 1)

        # Metrics to evaluate
        silhouette_scores = []
        ch_scores = []
        db_scores = []
        inertia_values = []

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_features)

            # Calculate metrics
            silhouette_scores.append(silhouette_score(self.scaled_features, labels))
            ch_scores.append(calinski_harabasz_score(self.scaled_features, labels))
            db_scores.append(davies_bouldin_score(self.scaled_features, labels))
            inertia_values.append(kmeans.inertia_)

        # Plot the metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Silhouette Score (higher is better)
        axes[0, 0].plot(k_values, silhouette_scores, 'bo-')
        axes[0, 0].set_xlabel('Number of clusters (k)')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Silhouette Score vs k (higher is better)')
        axes[0, 0].grid(True)

        # Calinski-Harabasz Index (higher is better)
        axes[0, 1].plot(k_values, ch_scores, 'ro-')
        axes[0, 1].set_xlabel('Number of clusters (k)')
        axes[0, 1].set_ylabel('Calinski-Harabasz Score')
        axes[0, 1].set_title('Calinski-Harabasz Score vs k (higher is better)')
        axes[0, 1].grid(True)

        # Davies-Bouldin Index (lower is better)
        axes[1, 0].plot(k_values, db_scores, 'go-')
        axes[1, 0].set_xlabel('Number of clusters (k)')
        axes[1, 0].set_ylabel('Davies-Bouldin Score')
        axes[1, 0].set_title('Davies-Bouldin Score vs k (lower is better)')
        axes[1, 0].grid(True)

        # Elbow Method (Inertia)
        axes[1, 1].plot(k_values, inertia_values, 'mo-')
        axes[1, 1].set_xlabel('Number of clusters (k)')
        axes[1, 1].set_ylabel('Inertia')
        axes[1, 1].set_title('Elbow Method (Inertia vs k)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('charts/clusters/optimal_k_metrics.png')
        plt.close()

        # Return metrics for different k values
        return {
            'k_values': list(k_values),
            'silhouette_scores': silhouette_scores,
            'calinski_harabasz_scores': ch_scores,
            'davies_bouldin_scores': db_scores,
            'inertia_values': inertia_values
        }

    def run_kmeans(self, k_values=None):
        """Run K-Means clustering with different k values."""
        if k_values is None:
            k_values = [3, 4, 5, 6]

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_features)

            # Store model and labels
            self.kmeans_models[k] = kmeans
            self.labels[k] = labels

            # Add cluster labels to dataframe
            self.df[f'cluster_k{k}'] = labels

        return self.labels

    def visualize_clusters(self, k_values=None):
        """Visualize clusters for different k values."""
        if k_values is None:
            k_values = list(self.kmeans_models.keys())

        # Run PCA for visualization
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(self.scaled_features)

        for k in k_values:
            if k not in self.labels:
                continue

            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1],
                       c=self.labels[k], cmap='viridis', alpha=0.7, s=100)
            plt.colorbar(scatter, label=f'Cluster label (k={k})')
            plt.title(f'K-Means Clustering with k={k} (PCA Projection)', fontsize=14)
            plt.xlabel('Principal Component 1', fontsize=12)
            plt.ylabel('Principal Component 2', fontsize=12)
            plt.grid(alpha=0.3)
            plt.savefig(f'charts/clusters/cluster_viz_k{k}.png')
            plt.close()

    def analyze_clusters(self, k):
        """Analyze clusters to extract insights."""
        if k not in self.labels:
            print(f"No clustering results for k={k}")
            return None

        cluster_stats = {}

        # Add cluster labels to the dataframe if not already there
        if f'cluster_k{k}' not in self.df.columns:
            self.df[f'cluster_k{k}'] = self.labels[k]

        # Calculate statistics for each cluster
        for cluster_id in range(k):
            cluster_df = self.df[self.df[f'cluster_k{k}'] == cluster_id]

            # Success rate
            success_rate = cluster_df['success'].mean() * 100

            # Most common skills
            current_skills = [skill for skills_list in cluster_df['current_skills'] for skill in skills_list]
            desired_skills = [skill for skills_list in cluster_df['desired_skills'] for skill in skills_list]
            target_skills = [skill for skills_list in cluster_df['target_skills'] for skill in skills_list]

            top_current = Counter(current_skills).most_common(5)
            top_desired = Counter(desired_skills).most_common(5)
            top_target = Counter(target_skills).most_common(5)

            # Skill gap statistics
            avg_gap = cluster_df['skill_gap_count'].mean()
            avg_efficiency = cluster_df['learning_efficiency'].mean()

            # Store statistics
            cluster_stats[cluster_id] = {
                'count': len(cluster_df),
                'success_rate': success_rate,
                'avg_skill_gap': avg_gap,
                'avg_learning_efficiency': avg_efficiency,
                'top_current_skills': top_current,
                'top_desired_skills': top_desired,
                'top_target_skills': top_target
            }

        # Visualize cluster statistics
        self._visualize_cluster_stats(k, cluster_stats)

        return cluster_stats

    def _visualize_cluster_stats(self, k, cluster_stats):
        """Create visualizations for cluster statistics."""
        # Plot success rate by cluster
        plt.figure(figsize=(14, 8))

        clusters = list(cluster_stats.keys())
        success_rates = [cluster_stats[cluster]['success_rate'] for cluster in clusters]
        counts = [cluster_stats[cluster]['count'] for cluster in clusters]

        # Size points by cluster population
        sizes = [count*100/sum(counts) for count in counts]

        # Create scatter plot of success rate vs. skill gap by cluster
        skill_gaps = [cluster_stats[cluster]['avg_skill_gap'] for cluster in clusters]
        efficiencies = [cluster_stats[cluster]['avg_learning_efficiency']*100 for cluster in clusters]

        plt.scatter(skill_gaps, success_rates, s=sizes, alpha=0.7, c=clusters, cmap='viridis')

        # Add cluster labels
        for i, cluster in enumerate(clusters):
            plt.annotate(f'Cluster {cluster}\n({counts[cluster]} learners)',
                        (skill_gaps[i], success_rates[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')

        plt.title(f'Success Rate vs. Skill Gap by Cluster (k={k})', fontsize=14)
        plt.xlabel('Average Skill Gap', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.colorbar(label='Cluster')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'charts/clusters/cluster_success_vs_gap_k{k}.png')
        plt.close()

        # Create skill distribution chart for top clusters
        fig, axes = plt.subplots(len(cluster_stats), 3, figsize=(18, 4*len(cluster_stats)))

        for i, cluster_id in enumerate(cluster_stats.keys()):
            # Current skills
            current_skills = dict(cluster_stats[cluster_id]['top_current_skills'])
            ax = axes[i, 0]
            ax.barh(list(current_skills.keys()), list(current_skills.values()), color='skyblue')
            ax.set_title(f'Cluster {cluster_id}: Current Skills')

            # Desired skills
            desired_skills = dict(cluster_stats[cluster_id]['top_desired_skills'])
            ax = axes[i, 1]
            ax.barh(list(desired_skills.keys()), list(desired_skills.values()), color='lightgreen')
            ax.set_title(f'Cluster {cluster_id}: Desired Skills')

            # Target skills
            target_skills = dict(cluster_stats[cluster_id]['top_target_skills'])
            ax = axes[i, 2]
            ax.barh(list(target_skills.keys()), list(target_skills.values()), color='salmon')
            ax.set_title(f'Cluster {cluster_id}: Target Skills')

        plt.tight_layout()
        plt.savefig(f'charts/clusters/cluster_skills_k{k}.png')
        plt.close()

    def identify_learning_paths(self, k):
        """Identify common learning paths within clusters."""
        if k not in self.labels:
            print(f"No clustering results for k={k}")
            return None

        learning_paths = {}

        for cluster_id in range(k):
            cluster_df = self.df[self.df[f'cluster_k{k}'] == cluster_id]

            # Get skill transitions (current → target)
            transitions = []
            for _, row in cluster_df.iterrows():
                current = set(row['current_skills'])
                target = set(row['target_skills'])

                # Focus on new skills being acquired
                new_skills = target - current

                if current and new_skills:  # Both have skills
                    for c_skill in current:
                        for t_skill in new_skills:
                            transitions.append((c_skill, t_skill))

            # Count transition frequencies
            transition_counts = Counter(transitions)
            common_paths = transition_counts.most_common(10)

            learning_paths[cluster_id] = common_paths

        # Visualize common learning paths
        self._visualize_learning_paths(k, learning_paths)

        return learning_paths

    def _visualize_learning_paths(self, k, learning_paths):
        """Visualize learning paths within clusters."""
        for cluster_id, paths in learning_paths.items():
            if not paths:
                continue

            plt.figure(figsize=(14, 10))

            # Create directed graph
            import networkx as nx
            G = nx.DiGraph()

            # Add edges with weights
            for (source, target), weight in paths:
                G.add_edge(source, target, weight=weight)

            # Set node positions
            pos = nx.spring_layout(G, k=0.3, seed=42)

            # Get edge weights for width
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            normalized_weights = [3 * w / max_weight for w in edge_weights]

            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue', alpha=0.8)
            nx.draw_networkx_labels(G, pos, font_size=10)
            nx.draw_networkx_edges(G, pos, width=normalized_weights,
                                  edge_color='gray', alpha=0.7, arrows=True,
                                  arrowstyle='->', arrowsize=15)

            # Add edge labels (weights)
            edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            plt.title(f'Learning Paths in Cluster {cluster_id} (k={k})', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'charts/clusters/learning_paths_cluster{cluster_id}_k{k}.png')
            plt.close()

    def evaluate_cluster_stability(self, k_values=None, n_samples=5):
        """Evaluate cluster stability using bootstrap sampling."""
        if k_values is None:
            k_values = [3, 4, 5]

        if self.scaled_features is None:
            print("Features not prepared. Call prepare_features() first.")
            return None

        # Results dictionary
        stability_results = {}

        for k in k_values:
            # Original clustering
            kmeans_original = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_original = kmeans_original.fit_predict(self.scaled_features)

            # Run multiple samplings
            sample_similarities = []

            for i in range(n_samples):
                # Bootstrap sample (with replacement)
                indices = np.random.choice(len(self.scaled_features), size=len(self.scaled_features), replace=True)
                sample_features = self.scaled_features[indices]

                # Cluster the sample
                kmeans_sample = KMeans(n_clusters=k, random_state=i, n_init=10)
                labels_sample = kmeans_sample.fit_predict(sample_features)

                # Map samples back to original indices
                mapped_labels = np.full(len(self.scaled_features), -1)
                mapped_labels[indices] = labels_sample

                # Compute similarity for overlapping points
                from sklearn.metrics import adjusted_rand_score

                # Only compare points that were selected in the bootstrap sample
                valid_indices = indices[indices < len(labels_original)]
                if len(valid_indices) > 0:
                    similarity = adjusted_rand_score(
                        labels_original[valid_indices],
                        labels_sample[:len(valid_indices)]
                    )
                    sample_similarities.append(similarity)

            # Store results
            stability_results[k] = {
                'mean_similarity': np.mean(sample_similarities),
                'std_similarity': np.std(sample_similarities),
                'similarities': sample_similarities
            }

        # Visualize stability
        self._visualize_cluster_stability(stability_results)

        return stability_results

    def _visualize_cluster_stability(self, stability_results):
        """Visualize cluster stability results."""
        k_values = sorted(stability_results.keys())
        means = [stability_results[k]['mean_similarity'] for k in k_values]
        stds = [stability_results[k]['std_similarity'] for k in k_values]

        plt.figure(figsize=(10, 6))
        plt.errorbar(k_values, means, yerr=stds, fmt='o-', capsize=5)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Adjusted Rand Index (stability)')
        plt.title('Cluster Stability Analysis')
        plt.grid(alpha=0.3)
        plt.ylim(0, 1)
        plt.savefig('charts/clusters/cluster_stability.png')
        plt.close()

    def recommend_skills_by_cluster(self, k):
        """Recommend next skills to learn for each cluster."""
        if k not in self.labels:
            print(f"No clustering results for k={k}")
            return None

        recommendations = {}

        # For each cluster
        for cluster_id in range(k):
            cluster_df = self.df[self.df[f'cluster_k{k}'] == cluster_id]

            # Get successful learners in this cluster
            successful_df = cluster_df[cluster_df['success'] == True]

            if len(successful_df) == 0:
                recommendations[cluster_id] = "No successful learners in this cluster"
                continue

            # Identify skills that successful people have learned
            successful_target_skills = [skill for skills_list in successful_df['target_skills']
                                       for skill in skills_list]
            successful_current_skills = [skill for skills_list in successful_df['current_skills']
                                       for skill in skills_list]

            # Find skills that were gained (in target but not in current)
            gained_skills = []
            for _, row in successful_df.iterrows():
                current = set(row['current_skills'])
                target = set(row['target_skills'])
                gained = target - current
                gained_skills.extend(gained)

            # Count frequencies
            gained_counts = Counter(gained_skills)
            top_recommendations = gained_counts.most_common(5)

            # Store recommendations
            recommendations[cluster_id] = {
                'top_recommendations': top_recommendations,
                'success_rate': len(successful_df) / len(cluster_df) * 100
            }

        # Visualize recommendations
        self._visualize_recommendations(k, recommendations)

        return recommendations

    def _visualize_recommendations(self, k, recommendations):
        """Visualize skill recommendations for each cluster."""
        clusters = sorted(recommendations.keys())

        plt.figure(figsize=(14, 8))

        for i, cluster_id in enumerate(clusters):
            # Check if we have actual recommendations
            if isinstance(recommendations[cluster_id], str):
                continue

            rec_dict = dict(recommendations[cluster_id]['top_recommendations'])
            if not rec_dict:
                continue

            skills = list(rec_dict.keys())
            counts = list(rec_dict.values())

            x = np.arange(len(skills))
            width = 0.8 / len(clusters)

            plt.bar(x + i*width, counts, width, label=f'Cluster {cluster_id}')

        plt.xlabel('Recommended Skills')
        plt.ylabel('Frequency in Successful Paths')
        plt.title(f'Top Skill Recommendations by Cluster (k={k})')
        plt.xticks([])  # Hide x-ticks as they would be overlapping
        plt.legend(title='Cluster')

        # Custom legend for skill names
        skill_legend = []
        for cluster_id in clusters:
            if isinstance(recommendations[cluster_id], str):
                continue

            for skill, count in recommendations[cluster_id]['top_recommendations']:
                if skill not in [item[0] for item in skill_legend]:
                    skill_legend.append((skill, count))

        # Add text box with skill names
        if skill_legend:
            skill_text = '\n'.join([f"{i+1}. {skill}" for i, (skill, _) in enumerate(skill_legend)])
            plt.figtext(0.85, 0.15, f"Skills:\n{skill_text}",
                      bbox=dict(facecolor='white', alpha=0.5), fontsize=8)

        plt.tight_layout()
        plt.savefig(f'charts/clusters/skill_recommendations_k{k}.png')
        plt.close()