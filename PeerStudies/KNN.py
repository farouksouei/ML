import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
import os


class PeerStudies:
    def __init__(self, data_processor=None, df=None):
        """
        Initialize PeerStudies module with either a DataProcessor instance or a pandas DataFrame.

        Parameters:
        -----------
        data_processor : DataProcessor, optional
            Instance of DataProcessor class containing processed data
        df : pandas.DataFrame, optional
            Processed DataFrame containing skills data
        """
        if data_processor is not None:
            self.df = data_processor.df
        elif df is not None:
            self.df = df
        else:
            raise ValueError("Either data_processor or df must be provided")

        # Create visualization directory if it doesn't exist
        os.makedirs('charts/peerStudies', exist_ok=True)

        # Initialize variables for later use
        self.feature_matrix = None
        self.skill_features = None
        self.knn_model = None
        self.peer_matches = None
        self.peer_groups = None

        # Check if data has been properly processed
        required_columns = ['current_skills', 'target_skills', 'skill_gap']
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError("Data must be processed with required columns: current_skills, target_skills, skill_gap")

    def extract_features(self, feature_types=None):
        """
        Extract features from the dataframe for KNN matching.

        Parameters:
        -----------
        feature_types : list, optional
            List of feature types to include ('current', 'target', 'gap', 'desired')
            If None, uses all current and target skills plus numeric features

        Returns:
        --------
        self : PeerStudies
            Returns self for method chaining
        """
        if feature_types is None:
            feature_types = ['current', 'target']

        # Get unique skills across all relevant columns
        all_skills = set()
        skill_cols = []

        if 'current' in feature_types:
            skill_cols.append('current_skills')
        if 'target' in feature_types:
            skill_cols.append('target_skills')
        if 'desired' in feature_types:
            skill_cols.append('desired_skills')
        if 'gap' in feature_types:
            skill_cols.append('skill_gap')

        for col in skill_cols:
            for skills in self.df[col]:
                all_skills.update(skills)

        # Create one-hot encoding for skills
        self.skill_features = pd.DataFrame(index=self.df.index)

        # One-hot encode current skills (what the user knows)
        if 'current' in feature_types:
            for skill in all_skills:
                self.skill_features[f'current_{skill}'] = self.df['current_skills'].apply(
                    lambda x: 1 if skill in x else 0
                )

        # One-hot encode target skills (what the user wants to learn)
        if 'target' in feature_types:
            for skill in all_skills:
                self.skill_features[f'target_{skill}'] = self.df['target_skills'].apply(
                    lambda x: 1 if skill in x else 0
                )

        # One-hot encode desired skills
        if 'desired' in feature_types:
            for skill in all_skills:
                self.skill_features[f'desired_{skill}'] = self.df['desired_skills'].apply(
                    lambda x: 1 if skill in x else 0
                )

        # One-hot encode skill gaps
        if 'gap' in feature_types:
            for skill in all_skills:
                self.skill_features[f'gap_{skill}'] = self.df['skill_gap'].apply(
                    lambda x: 1 if skill in x else 0
                )

        if all(col in self.df.columns for col in ['skill_count_current', 'skill_count_target']):
            numeric_cols = ['skill_count_current', 'skill_count_target']
            if 'skill_gap_count' in self.df.columns:
                numeric_cols.append('skill_gap_count')
            if 'learning_efficiency' in self.df.columns:
                numeric_cols.append('learning_efficiency')

            numeric_features = self.df[numeric_cols].copy()

            # Scale numeric features
            scaler = StandardScaler()
            numeric_features_scaled = pd.DataFrame(
                scaler.fit_transform(numeric_features),
                index=numeric_features.index,
                columns=numeric_features.columns
            )
            self.feature_matrix = pd.concat([self.skill_features, numeric_features_scaled], axis=1)
        else:
            self.feature_matrix = self.skill_features

        print(f"Created feature matrix with {self.feature_matrix.shape[1]} dimensions")
        return self

    def fit_knn(self, n_neighbors=5, algorithm='auto', metric='jaccard'):
        """
        Fit KNN model to the feature matrix.

        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to consider for each user
        algorithm : str, default='auto'
            Algorithm used to compute nearest neighbors
        metric : str, default='jaccard'
            Distance metric to use for matching (jaccard works well for binary features)

        Returns:
        --------
        self : PeerStudies
            Returns self for method chaining
        """
        if self.feature_matrix is None:
            self.extract_features()

        # Create and fit KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # +1 because the user itself is included
            algorithm=algorithm,
            metric=metric
        )
        self.knn_model.fit(self.feature_matrix)

        return self

    def find_peer_matches(self, visualize=True):
        """
        Find peer matches for all users based on their feature similarity.

        Parameters:
        -----------
        visualize : bool, default=True
            Whether to create visualizations of peer matches

        Returns:
        --------
        dict
            Dictionary containing peer match information for each user
        """
        if self.knn_model is None:
            self.fit_knn()

        # Get distances and indices of nearest neighbors
        distances, indices = self.knn_model.kneighbors(self.feature_matrix)

        # Create a dictionary to store peer match information
        self.peer_matches = {}

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            user_id = self.df.iloc[i]['id']

            # Skip the first match (which is the user itself)
            peer_info = []
            for j in range(1, len(idx)):
                peer_idx = idx[j]
                peer_id = self.df.iloc[peer_idx]['id']

                # Calculate complementary skills (what peers can teach each other)
                user_skills = set(self.df.iloc[i]['current_skills'])
                peer_skills = set(self.df.iloc[peer_idx]['current_skills'])
                user_gaps = set(self.df.iloc[i]['skill_gap'])
                peer_gaps = set(self.df.iloc[peer_idx]['skill_gap'])

                # What peer can teach user
                peer_can_teach = [skill for skill in user_gaps if skill in peer_skills]

                # What user can teach peer
                user_can_teach = [skill for skill in peer_gaps if skill in user_skills]

                # Calculate match quality score
                # Higher score means better teaching match
                if len(user_gaps) > 0 and len(peer_gaps) > 0:
                    reciprocal_score = (len(peer_can_teach) / len(user_gaps)) * \
                                       (len(user_can_teach) / len(peer_gaps))
                else:
                    reciprocal_score = 0

                # Overall compatibility - lower distance means better match from KNN
                # Normalize distance to 0-1 scale and invert
                normalized_distance = 1 - (dist[j] / max(dist[1:]))

                # Combined score (weighted average)
                # Emphasize reciprocal learning (0.7) over general similarity (0.3)
                match_score = (0.7 * reciprocal_score) + (0.3 * normalized_distance)

                peer_info.append({
                    'peer_id': peer_id,
                    'distance': dist[j],
                    'peer_can_teach': peer_can_teach,
                    'user_can_teach': user_can_teach,
                    'reciprocal_score': reciprocal_score,
                    'similarity_score': normalized_distance,
                    'match_score': match_score
                })

            # Sort peers by match score
            sorted_peers = sorted(peer_info, key=lambda x: x['match_score'], reverse=True)

            self.peer_matches[user_id] = {
                'user_id': user_id,
                'current_skills': self.df.iloc[i]['current_skills'],
                'target_skills': self.df.iloc[i]['target_skills'],
                'skill_gap': self.df.iloc[i]['skill_gap'],
                'peer_matches': sorted_peers
            }

        if visualize:
            self._visualize_peer_network()

        return self.peer_matches

    def form_study_groups(self, max_group_size=4, method='greedy', visualize=True):
        """
        Form study groups based on peer matches.

        Parameters:
        -----------
        max_group_size : int, default=4
            Maximum number of students in each study group
        method : str, default='greedy'
            Method to use for forming groups ('greedy' or 'optimal')
        visualize : bool, default=True
            Whether to create visualizations of study groups

        Returns:
        --------
        list
            List of study groups with member details
        """
        if self.peer_matches is None:
            self.find_peer_matches(visualize=False)

        if method == 'greedy':
            self.peer_groups = self._form_groups_greedy(max_group_size)
        elif method == 'optimal':
            self.peer_groups = self._form_groups_optimal(max_group_size)
        else:
            raise ValueError("Method must be either 'greedy' or 'optimal'")

        if visualize:
            self._visualize_study_groups()

        return self.peer_groups

    def _form_groups_greedy(self, max_group_size=4):
        """
        Form study groups using a greedy algorithm.

        Parameters:
        -----------
        max_group_size : int, default=4
            Maximum number of students in each study group

        Returns:
        --------
        list
            List of study groups with member details
        """
        # Create a list of all users
        all_users = list(self.peer_matches.keys())
        user_assigned = {user_id: False for user_id in all_users}

        # Initialize groups
        groups = []

        # Keep forming groups until all users are assigned
        while False in user_assigned.values():
            # Find an unassigned user
            seed_user = next(uid for uid, assigned in user_assigned.items() if not assigned)
            user_assigned[seed_user] = True

            # Create a new group with the seed user
            new_group = {
                'group_id': len(groups),
                'members': [{'user_id': seed_user,
                             'current_skills': self.peer_matches[seed_user]['current_skills'],
                             'target_skills': self.peer_matches[seed_user]['target_skills']}],
                'skill_coverage': set(self.peer_matches[seed_user]['current_skills'])
            }

            # Add best matching peers to the group
            for peer_match in self.peer_matches[seed_user]['peer_matches']:
                peer_id = peer_match['peer_id']

                if not user_assigned[peer_id] and len(new_group['members']) < max_group_size:
                    user_assigned[peer_id] = True
                    new_group['members'].append({
                        'user_id': peer_id,
                        'current_skills': self.peer_matches[peer_id]['current_skills'],
                        'target_skills': self.peer_matches[peer_id]['target_skills']
                    })
                    new_group['skill_coverage'].update(self.peer_matches[peer_id]['current_skills'])

                    if len(new_group['members']) >= max_group_size:
                        break

            # Calculate group metrics
            new_group['skill_coverage'] = list(new_group['skill_coverage'])
            new_group['coverage_count'] = len(new_group['skill_coverage'])

            # Analyze teaching potential within group
            all_gaps = []
            covered_gaps = []

            for member in new_group['members']:
                user_id = member['user_id']
                gaps = self.peer_matches[user_id]['skill_gap']
                all_gaps.extend(gaps)

                # Check which gaps can be covered by other group members
                for gap_skill in gaps:
                    if gap_skill in new_group['skill_coverage']:
                        covered_gaps.append(gap_skill)

            new_group['gaps_covered_count'] = len(covered_gaps)
            new_group['total_gaps_count'] = len(all_gaps)

            if len(all_gaps) > 0:
                new_group['coverage_ratio'] = len(covered_gaps) / len(all_gaps)
            else:
                new_group['coverage_ratio'] = 0

            groups.append(new_group)

        return groups

    def _form_groups_optimal(self, max_group_size=4):
        """
        Form study groups using a more sophisticated algorithm that tries to optimize
        overall learning potential across all groups.

        Parameters:
        -----------
        max_group_size : int, default=4
            Maximum number of students in each study group

        Returns:
        --------
        list
            List of study groups with member details
        """
        import networkx as nx

        # Create a graph where nodes are users and edges represent match scores
        G = nx.Graph()

        # Add all users as nodes
        for user_id in self.peer_matches:
            G.add_node(user_id)

        # Add edges between users based on match scores
        for user_id, user_data in self.peer_matches.items():
            for peer_match in user_data['peer_matches']:
                peer_id = peer_match['peer_id']
                match_score = peer_match['match_score']

                # Add edge if it doesn't exist or update if the new score is higher
                if not G.has_edge(user_id, peer_id) or G[user_id][peer_id]['weight'] < match_score:
                    G.add_edge(user_id, peer_id, weight=match_score)

        # Use community detection to find groups
        # This helps find naturally forming clusters in the peer network
        try:
            from community import best_partition
            partition = best_partition(G, weight='weight')
        except ImportError:
            # Fallback to greedy algorithm if community detection is not available
            print("Community detection package not found. Using modular approach.")
            communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i

        # Map partition to groups
        community_to_users = {}
        for user_id, community_id in partition.items():
            if community_id not in community_to_users:
                community_to_users[community_id] = []
            community_to_users[community_id].append(user_id)

        # Split large communities into groups of max_group_size
        groups = []
        group_id = 0

        for community_id, members in community_to_users.items():
            # Sort members by their connectedness within community
            member_scores = []
            for user_id in members:
                # Calculate average match score with other community members
                conn_scores = []
                for other_id in members:
                    if user_id != other_id and G.has_edge(user_id, other_id):
                        conn_scores.append(G[user_id][other_id]['weight'])

                avg_score = sum(conn_scores) / len(conn_scores) if conn_scores else 0
                member_scores.append((user_id, avg_score))

            # Sort by score descending
            sorted_members = [x[0] for x in sorted(member_scores, key=lambda x: x[1], reverse=True)]

            # Split into groups of max_group_size
            for i in range(0, len(sorted_members), max_group_size):
                group_members = sorted_members[i:i + max_group_size]

                # Create group
                new_group = {
                    'group_id': group_id,
                    'members': [],
                    'skill_coverage': set()
                }

                for user_id in group_members:
                    new_group['members'].append({
                        'user_id': user_id,
                        'current_skills': self.peer_matches[user_id]['current_skills'],
                        'target_skills': self.peer_matches[user_id]['target_skills']
                    })
                    new_group['skill_coverage'].update(self.peer_matches[user_id]['current_skills'])

                # Calculate group metrics
                new_group['skill_coverage'] = list(new_group['skill_coverage'])
                new_group['coverage_count'] = len(new_group['skill_coverage'])

                # Analyze teaching potential within group
                all_gaps = []
                covered_gaps = []

                for member in new_group['members']:
                    user_id = member['user_id']
                    gaps = self.peer_matches[user_id]['skill_gap']
                    all_gaps.extend(gaps)

                    # Check which gaps can be covered by other group members
                    for gap_skill in gaps:
                        if gap_skill in new_group['skill_coverage']:
                            covered_gaps.append(gap_skill)

                new_group['gaps_covered_count'] = len(covered_gaps)
                new_group['total_gaps_count'] = len(all_gaps)

                if len(all_gaps) > 0:
                    new_group['coverage_ratio'] = len(covered_gaps) / len(all_gaps)
                else:
                    new_group['coverage_ratio'] = 0

                groups.append(new_group)
                group_id += 1

        return groups

    def _visualize_peer_network(self):
        """Visualize peer matching as a network."""
        G = nx.Graph()

        # Add nodes for each user
        for user_id in self.peer_matches:
            G.add_node(user_id)

        # Add edges for top matches
        for user_id, data in self.peer_matches.items():
            # Add edges to top 3 peers or fewer if not enough matches
            top_peers = data['peer_matches'][:min(3, len(data['peer_matches']))]
            for peer in top_peers:
                G.add_edge(user_id, peer['peer_id'], weight=peer['match_score'])

        # Sample a subset of the graph if it's too large
        if len(G.nodes()) > 50:
            sampled_nodes = np.random.choice(list(G.nodes()), size=50, replace=False)
            G = G.subgraph(sampled_nodes)

        plt.figure(figsize=(15, 15))

        # Layout
        pos = nx.spring_layout(G, k=0.3, seed=42)

        # Get edge weights for line thickness
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]

        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=10)

        plt.title('Peer Matching Network', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('charts/peerStudies/peer_network.png')
        plt.close()

        # Create 2D visualization of user similarity
        self._visualize_user_similarity()

    def _visualize_user_similarity(self):
        """Create a 2D visualization of user similarity using PCA."""
        # Apply PCA to reduce features to 2D for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.feature_matrix)

        # Create a dataframe for plotting
        vis_df = pd.DataFrame({
            'user_id': self.df['id'],
            'x': features_2d[:, 0],
            'y': features_2d[:, 1],
            'success': self.df['success'] if 'success' in self.df.columns else None
        })

        plt.figure(figsize=(12, 10))

        # Color by success if available
        if 'success' in self.df.columns:
            sns.scatterplot(
                data=vis_df,
                x='x',
                y='y',
                hue='success',
                palette=['red', 'green'],
                s=100,
                alpha=0.7
            )
            plt.legend(title='Success')
        else:
            sns.scatterplot(
                data=vis_df,
                x='x',
                y='y',
                s=100,
                alpha=0.7
            )

        # Add user IDs as labels
        for i, row in vis_df.iterrows():
            plt.annotate(
                str(row['user_id']),
                (row['x'], row['y']),
                fontsize=8,
                alpha=0.8
            )

        # Add explained variance as title
        explained_var = pca.explained_variance_ratio_ * 100
        plt.title(f'User Similarity Map (PCA) - {explained_var[0]:.1f}% and {explained_var[1]:.1f}% variance',
                  fontsize=14)
        plt.xlabel(f'Principal Component 1')
        plt.ylabel(f'Principal Component 2')
        plt.tight_layout()
        plt.savefig('charts/peerStudies/user_similarity_map.png')
        plt.close()

    def _visualize_study_groups(self):
        """Visualize formed study groups."""
        if not self.peer_groups:
            return

        # Create group visualization
        plt.figure(figsize=(14, 8))

        # Group size distribution
        group_sizes = [len(group['members']) for group in self.peer_groups]
        plt.subplot(1, 2, 1)
        sns.countplot(x=group_sizes)
        plt.title('Group Size Distribution')
        plt.xlabel('Group Size')
        plt.ylabel('Count')

        # Coverage ratio distribution
        coverage_ratios = [group['coverage_ratio'] for group in self.peer_groups]
        plt.subplot(1, 2, 2)
        sns.histplot(coverage_ratios, bins=10, kde=True)
        plt.title('Skill Coverage Ratio Distribution')
        plt.xlabel('Coverage Ratio (higher is better)')

        plt.tight_layout()
        plt.savefig('charts/peerStudies/group_stats.png')
        plt.close()

        # Group skill coverage visualization (top 5 groups)
        top_groups = sorted(self.peer_groups, key=lambda x: x['coverage_ratio'], reverse=True)[:5]

        plt.figure(figsize=(16, 12))

        for i, group in enumerate(top_groups, 1):
            plt.subplot(3, 2, i)

            # Count member skills and gaps
            member_skills = {}
            member_gaps = {}

            for member in group['members']:
                for skill in member['current_skills']:
                    if skill not in member_skills:
                        member_skills[skill] = 0
                    member_skills[skill] += 1

                user_id = member['user_id']
                for gap in self.peer_matches[user_id]['skill_gap']:
                    if gap not in member_gaps:
                        member_gaps[gap] = 0
                    member_gaps[gap] += 1

            # Plot top skills and gaps
            top_skills = dict(sorted(member_skills.items(), key=lambda x: x[1], reverse=True)[:10])
            df = pd.DataFrame({
                'Skill': list(top_skills.keys()),
                'Count': list(top_skills.values())
            })

            sns.barplot(data=df, x='Count', y='Skill', palette='viridis')
            plt.title(f'Group {group["group_id"]} Skills (CR: {group["coverage_ratio"]:.2f})')
            plt.xlabel('Number of Members')
            plt.tight_layout()

        plt.savefig('charts/peerStudies/top_groups_skills.png')
        plt.close()

        # Create network visualization of top group
        if top_groups:
            self._visualize_group_network(top_groups[0])

    def _visualize_group_network(self, group):
        """Visualize the internal network of a study group."""
        G = nx.Graph()

        # Add nodes for each member
        for member in group['members']:
            user_id = member['user_id']
            G.add_node(user_id)

        # Add edges between members who can teach each other
        for i, member1 in enumerate(group['members']):
            user1_id = member1['user_id']
            user1_current = set(member1['current_skills'])

            for j, member2 in enumerate(group['members']):
                if i >= j:  # Avoid duplicate edges and self-loops
                    continue

                user2_id = member2['user_id']
                user2_current = set(member2['current_skills'])

                # What user1 can teach user2
                user1_can_teach = [skill for skill in self.peer_matches[user2_id]['skill_gap']
                                   if skill in user1_current]

                # What user2 can teach user1
                user2_can_teach = [skill for skill in self.peer_matches[user1_id]['skill_gap']
                                   if skill in user2_current]

                # Add edge with teaching info
                if user1_can_teach or user2_can_teach:
                    G.add_edge(
                        user1_id,
                        user2_id,
                        user1_teaches=user1_can_teach,
                        user2_teaches=user2_can_teach,
                        weight=len(user1_can_teach) + len(user2_can_teach)
                    )

        plt.figure(figsize=(12, 12))

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.8)

        # Draw edges with thickness based on teaching potential
        edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='gray')

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12)

        # Add edge labels - what skills are taught
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            u_teaches = data.get('user1_teaches', [])
            v_teaches = data.get('user2_teaches', [])

            if len(u_teaches) <= 2 and len(v_teaches) <= 2:
                u_skills = ', '.join(u_teaches) if u_teaches else ""
                v_skills = ', '.join(v_teaches) if v_teaches else ""
                edge_labels[(u, v)] = f"{u} → {v}: {u_skills}\n{v} → {u}: {v_skills}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title(f'Group {group["group_id"]} Teaching Network', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('charts/peerStudies/group_network.png')
        plt.close()

    def get_best_peer_match(self, user_id, top_n=3):
        """
        Get the best peer matches for a specific user.

        Parameters:
        -----------
        user_id : int or str
            ID of the user to find matches for
        top_n : int, default=3
            Number of top matches to return

        Returns:
        --------
        dict
            Dictionary containing user information and top peer matches
        """
        if self.peer_matches is None:
            self.find_peer_matches(visualize=False)

        if user_id not in self.peer_matches:
            return {'error': f"User ID {user_id} not found in peer matches."}

        user_matches = self.peer_matches[user_id]
        top_matches = user_matches['peer_matches'][:top_n]

        result = {
            'user_id': user_id,
            'current_skills': user_matches['current_skills'],
            'target_skills': user_matches['target_skills'],
            'skill_gap': user_matches['skill_gap'],
            'top_matches': []
        }

        for match in top_matches:
            peer_id = match['peer_id']
            peer_data = self.df[self.df['id'] == peer_id].iloc[0]

            result['top_matches'].append({
                'peer_id': peer_id,
                'match_score': match['match_score'],
                'skills_peer_can_teach': match['peer_can_teach'],
                'skills_user_can_teach': match['user_can_teach'],
                'reciprocal_learning_score': match['reciprocal_score']
            })

        return result

    def recommend_study_peer(self, skills, desired_skills, top_n=1):
        """
        Recommend the best peer for a user based on input skills and desired skills.

        Parameters:
        -----------
        skills : list
            List of the user's current skills
        desired_skills : list
            List of skills the user wants to learn
        top_n : int, default=1
            Number of top matches to return (default is 1 for best match)

        Returns:
        --------
        dict
            Dictionary containing the recommended peer(s) and their details
        """
        # Create a temporary dataframe for the new user
        new_user = pd.DataFrame([{
            'id': 'new_user',
            'current_skills': skills,
            'target_skills': desired_skills,
            'skill_gap': [skill for skill in desired_skills if skill not in skills]
        }])

        # Append the new user to the current dataframe
        temp_df = pd.concat([self.df, new_user], ignore_index=True)

        # Reinitialize the PeerStudies instance with the new dataframe
        temp_peer_studies = PeerStudies(df=temp_df)
        temp_peer_studies.extract_features()
        temp_peer_studies.fit_knn()
        temp_peer_studies.find_peer_matches(visualize=False)

        # Get matches for the new user
        new_user_matches = temp_peer_studies.peer_matches['new_user']['peer_matches'][:top_n]
        recommendations = []

        for match in new_user_matches:
            peer_id = match['peer_id']
            peer_data = self.df[self.df['id'] == peer_id].iloc[0]

            recommendations.append({
                'peer_id': peer_id,
                'peer_skills': peer_data['current_skills'],
                'peer_target_skills': peer_data['target_skills'],
                'match_score': match['match_score'],
                'skills_peer_can_teach': match['peer_can_teach'],
                'skills_user_can_teach': match['user_can_teach']
            })

        return recommendations