import pandas as pd
from DataProcessor import DataProcessor
from KMeansSkillAnalysis import KMeansSkillAnalysis

def main():
    # Load and process data
    processor = DataProcessor(file_path='../data/edited_skill_exchange_dataset.csv', columns=6)
    processor.clean_data()

    # Initialize K-means analysis
    km_analysis = KMeansSkillAnalysis(processor)

    # Prepare features for clustering
    features = km_analysis.prepare_features()
    print(f"Prepared {features.shape[1]} features for clustering")

    # Find optimal k
    print("Finding optimal number of clusters...")
    metrics = km_analysis.find_optimal_k(max_k=10)

    # Run K-means with multiple k values
    k_values = [3, 4, 5, 6]
    print(f"Running K-means with k values: {k_values}")
    km_analysis.run_kmeans(k_values)

    # Visualize clusters
    print("Visualizing clusters...")
    km_analysis.visualize_clusters()

    # Analyze the best k value (assumed to be 4 for this example)
    best_k = 4
    print(f"Analyzing clusters for k={best_k}...")
    cluster_stats = km_analysis.analyze_clusters(best_k)

    # Identify learning paths
    print("Identifying learning paths...")
    learning_paths = km_analysis.identify_learning_paths(best_k)

    # Evaluate cluster stability
    print("Evaluating cluster stability...")
    stability = km_analysis.evaluate_cluster_stability()

    # Generate skill recommendations
    print("Generating skill recommendations by cluster...")
    recommendations = km_analysis.recommend_skills_by_cluster(best_k)

    # Print summary
    print("\nAnalysis Summary:")
    print(f"Number of instances: {len(processor.df)}")
    print(f"Optimal cluster count (suggested): {best_k}")

    for cluster_id, stats in cluster_stats.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Population: {stats['count']} learners ({stats['count']/len(processor.df)*100:.1f}%)")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Avg skill gap: {stats['avg_skill_gap']:.1f}")
        print(f"  Avg learning efficiency: {stats['avg_learning_efficiency']:.2f}")

        print("  Top current skills:", ", ".join([skill for skill, _ in stats['top_current_skills'][:3]]))
        print("  Top target skills:", ", ".join([skill for skill, _ in stats['top_target_skills'][:3]]))

        if cluster_id in recommendations:
            if isinstance(recommendations[cluster_id], str):
                print(f"  Recommendations: {recommendations[cluster_id]}")
            else:
                rec_skills = [skill for skill, _ in recommendations[cluster_id]['top_recommendations']]
                print(f"  Recommended skills: {', '.join(rec_skills)}")

if __name__ == "__main__":
    main()