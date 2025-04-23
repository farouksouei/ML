import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from DataProcessor import DataProcessor
from PeerStudies.KNN import PeerStudies
from PeerStudies.testSVM import test_peer_recommendationsSVM


def test_knn_basics():
    """Test basic KNN functionality with data processing pipeline."""
    print("\n=== Testing Basic KNN Functionality ===")

    # Initialize processor with file path - pass the correct number of columns
    processor = DataProcessor(file_path='data/edited_skill_exchange_dataset.csv', columns=6)

    # Print the original column names to debug
    print(f"Original columns in dataset: {processor.df.columns.tolist()}")

    # Apply correct column names for your data
    processor.df.columns = ["id", "enrollment_date", "current_skills", "desired_skills", "target_skills", "success"]
    print(f"Renamed columns: {processor.df.columns.tolist()}")

    # Clean and process the data
    processor.clean_data()

    print(f"Data shape after processing: {processor.df.shape}")

    # Initialize PeerStudies with processed data
    peer_studies = PeerStudies(data_processor=processor)

    # Test feature extraction
    peer_studies.extract_features()
    print(f"Feature matrix shape: {peer_studies.feature_matrix.shape}")

    # Test KNN model fitting
    peer_studies.fit_knn(n_neighbors=5)
    print("KNN model successfully fitted")

    # Test peer matching
    peer_matches = peer_studies.find_peer_matches(visualize=True)
    sample_user_id = processor.df.iloc[0]['id']
    print(f"Sample peer match for user {sample_user_id}:")

    # Print top match details
    top_match = peer_studies.get_best_peer_match(sample_user_id, top_n=1)
    print(f"Top match ID: {top_match['top_matches'][0]['peer_id']}")
    print(f"Match score: {top_match['top_matches'][0]['match_score']:.3f}")
    print(f"User can learn: {top_match['top_matches'][0]['skills_peer_can_teach']}")
    print(f"User can teach: {top_match['top_matches'][0]['skills_user_can_teach']}")

    return peer_studies


def test_study_groups(peer_studies):
    """Test study group formation."""
    print("\n=== Testing Study Group Formation ===")

    # Test greedy group formation
    greedy_groups = peer_studies.form_study_groups(max_group_size=4, method='greedy')
    print(f"Created {len(greedy_groups)} greedy groups")

    # Sample information from first group
    if greedy_groups:
        first_group = greedy_groups[0]
        print(f"First group size: {len(first_group['members'])}")
        print(f"Skill coverage: {first_group['coverage_count']} skills")
        print(f"Coverage ratio: {first_group['coverage_ratio']:.2f}")

    # Test optimal group formation if available
    try:
        optimal_groups = peer_studies.form_study_groups(max_group_size=4, method='optimal')
        print(f"Created {len(optimal_groups)} optimal groups")
    except ImportError:
        print("Optimal grouping requires additional libraries")

    return greedy_groups


def test_peer_recommendations():
    """Test peer recommendation for new users."""
    print("\n=== Testing Peer Recommendations ===")

    # Initialize with file path and correct column count
    processor = DataProcessor(file_path='data/edited_skill_exchange_dataset.csv', columns=6)
    processor.df.columns = ["id", "enrollment_date", "current_skills", "desired_skills", "target_skills", "success"]
    processor.clean_data()

    # Initialize PeerStudies with processed data
    peer_studies = PeerStudies(data_processor=processor)
    peer_studies.extract_features()
    peer_studies.fit_knn()

    # Test cases with different skill combinations
    test_cases = [
        {
            "name": "Beginner Developer",
            "current_skills": ["HTML", "CSS", "JavaScript"],
            "desired_skills": ["React", "Node.js", "MongoDB"]
        },
        {
            "name": "Data Scientist",
            "current_skills": ["Python", "Statistics", "Machine Learning"],
            "desired_skills": ["Deep Learning", "NLP", "Computer Vision"]
        },
        {
            "name": "DevOps Engineer",
            "current_skills": ["Linux", "Docker", "AWS"],
            "desired_skills": ["Kubernetes", "Terraform", "CI/CD"]
        }
    ]

    for case in test_cases:
        print(f"\nTesting recommendation for {case['name']}:")
        print(f"Skills: {', '.join(case['current_skills'])}")
        print(f"Wants to learn: {', '.join(case['desired_skills'])}")

        try:
            # Create a temporary DataFrame in the correct format
            temp_df = pd.DataFrame([{
                'id': 'new_user',
                'current_skills': case['current_skills'],
                'desired_skills': case['desired_skills'],
                'target_skills': case['desired_skills'],  # Set target_skills same as desired_skills
                'success': False,  # Default value
                'enrollment_date': datetime.now().strftime("%Y-%m-%d"),  # Current date
                'skill_count_current': len(case['current_skills']),
                'skill_count_desired': len(case['desired_skills']),
                'skill_count_target': len(case['desired_skills']),
                'skill_gap': [skill for skill in case['desired_skills'] if skill not in case['current_skills']],
                'skill_gap_count': sum(1 for skill in case['desired_skills'] if skill not in case['current_skills']),
                'skill_overlap': [skill for skill in case['desired_skills'] if skill in case['current_skills']],
                'skill_overlap_count': sum(1 for skill in case['desired_skills'] if skill in case['current_skills']),
                'learning_efficiency': 0.0  # Default value
            }])

            # Directly get recommendations by matching against existing users
            matching_users = []
            for idx, row in peer_studies.df.iterrows():
                peer_id = row['id']
                peer_skills = set(row['current_skills'])
                user_skills = set(case['current_skills'])
                desired_skills = set(case['desired_skills'])

                # What peer can teach user
                peer_can_teach = [skill for skill in desired_skills if skill in peer_skills]

                # What user can teach peer
                peer_gaps = [skill for skill in row['target_skills'] if skill not in peer_skills]
                user_can_teach = [skill for skill in peer_gaps if skill in user_skills]

                # Calculate match score
                teaching_score = len(peer_can_teach) / max(1, len(desired_skills))
                reciprocal_score = len(user_can_teach) / max(1, len(peer_gaps)) if peer_gaps else 0
                match_score = (0.7 * teaching_score) + (0.3 * reciprocal_score)

                matching_users.append({
                    'peer_id': peer_id,
                    'match_score': match_score,
                    'skills_peer_can_teach': peer_can_teach,
                    'skills_user_can_teach': user_can_teach
                })

            # Sort by match score and get top match
            matching_users.sort(key=lambda x: x['match_score'], reverse=True)
            recommendation = matching_users[:1] if matching_users else []

            if recommendation:
                r = recommendation[0]
                print(f"Recommended peer ID: {r['peer_id']}")
                print(f"Match score: {r['match_score']:.3f}")
                print(f"Peer can teach: {', '.join(r['skills_peer_can_teach'])}")
                print(f"User can teach peer: {', '.join(r['skills_user_can_teach'])}")
            else:
                print("No recommendation found")

        except Exception as e:
            print(f"Error recommending for {case['name']}: {e}")

def test_skill_analysis(peer_studies):
    """Analyze skill relationships and gaps."""
    print("\n=== Testing Skill Analysis ===")

    # Get unique skills
    all_skills = set()
    for col in ['current_skills', 'target_skills']:
        for skills_list in peer_studies.df[col]:
            all_skills.update(skills_list)

    print(f"Total unique skills in dataset: {len(all_skills)}")

    # Most common skills
    all_current_skills = []
    for skills in peer_studies.df['current_skills']:
        all_current_skills.extend(skills)

    skill_counts = pd.Series(all_current_skills).value_counts().head(5)
    print("\nTop 5 most common skills:")
    for skill, count in skill_counts.items():
        print(f"{skill}: {count}")

    # Most desired skills
    all_target_skills = []
    for skills in peer_studies.df['target_skills']:
        all_target_skills.extend(skills)

    target_counts = pd.Series(all_target_skills).value_counts().head(5)
    print("\nTop 5 most desired skills:")
    for skill, count in target_counts.items():
        print(f"{skill}: {count}")


def visualize_skill_pairings(peer_studies):
    """Visualize common skill pairings in the dataset."""
    print("\n=== Visualizing Skill Pairings ===")

    # Create a chart directory if it doesn't exist
    os.makedirs('charts', exist_ok=True)

    # Analyze skill co-occurrences in current skills
    skill_pairs = {}

    for skills_list in peer_studies.df['current_skills']:
        if len(skills_list) < 2:
            continue

        for i, skill1 in enumerate(skills_list):
            for skill2 in skills_list[i + 1:]:
                pair = tuple(sorted([skill1, skill2]))
                if pair not in skill_pairs:
                    skill_pairs[pair] = 0
                skill_pairs[pair] += 1

    # Get top pairs
    top_pairs = sorted(skill_pairs.items(), key=lambda x: x[1], reverse=True)[:15]

    # Create DataFrame for plotting
    pair_df = pd.DataFrame({
        'Skill Pair': [f"{pair[0][0]} & {pair[0][1]}" for pair in top_pairs],
        'Count': [pair[1] for pair in top_pairs]
    })

    # Plot
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Count', y='Skill Pair', data=pair_df)
    plt.title('Most Common Skill Pairings in Current Skills', fontsize=14)
    plt.xlabel('Number of Users', fontsize=12)
    plt.tight_layout()
    plt.savefig('charts/skill_pairings.png')
    plt.close()

    print(f"Visualization saved to charts/skill_pairings.png")


def main():
    """Main test function."""
    print("Starting KNN Testing Suite")
    print("=" * 50)

    # Basic KNN functionality test
    try:
        peer_studies = test_knn_basics()

        # Only continue with other tests if first test passes
        # Study group formation test
        study_groups = test_study_groups(peer_studies)

        # Test peer recommendations for new users
        test_peer_recommendations()

        # Analyze skills in the dataset
        test_skill_analysis(peer_studies)

        # Create visualizations
        visualize_skill_pairings(peer_studies)

        # test svm
        test_peer_recommendationsSVM()


        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()