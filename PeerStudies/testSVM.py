from DataProcessor import DataProcessor
from PeerStudies.SVM import PeerStudiesSVM


def test_peer_recommendationsSVM():
    """Test peer recommendation for new users."""
    print("\n=== Testing Peer Recommendations ===")

    # Initialize with file path and correct column count
    processor = DataProcessor(file_path='data/edited_skill_exchange_dataset.csv', columns=6)
    processor.df.columns = ["id", "enrollment_date", "current_skills", "desired_skills", "target_skills", "success"]
    processor.clean_data()

    # Initialize PeerStudies with processed data
    peer_studies = PeerStudiesSVM(data_processor=processor)

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
            # Directly match against existing users without KNN
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