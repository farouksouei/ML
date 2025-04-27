import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import networkx as nx


class DataProcessor:
    def __init__(self, file_path=None, data=None, columns=None):
        """Initialize DataProcessor with either file path or data."""
        self.columns = columns
        if file_path:
            self.load_data_from_file(file_path)
        elif data is not None:
            self.df = pd.DataFrame(data)
        else:
            self.df = pd.DataFrame()

        # Create visualization directory
        os.makedirs('charts/dataViz', exist_ok=True)

    def load_data_from_file(self, file_path):
        """Load data from CSV file."""
        try:
            self.df = pd.read_csv(file_path, header=None)
            # Only set column names if they're not already set
            if not any(isinstance(col, str) for col in self.df.columns):
                self.set_column_names()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = pd.DataFrame()

    def set_column_names(self):
        """Set column names for the dataframe."""
        # Use different column names based on the number of columns
        if self.columns == 6:
            column_names = ["id", "enrollment_date", "current_skills", "desired_skills", "target_skills", "success"]
        else:
            column_names = ["id", "current_skills", "desired_skills", "target_skills", "success"]

        try:
            self.df.columns = column_names
        except ValueError as e:
            print(f"Column mismatch: {e}")
            print(f"DataFrame has {self.df.shape[1]} columns but {len(column_names)} names were provided")

    def print_and_save_sample(self, prefix="", sample_size=5):
        """Print a sample of the data and save it to a file."""
        print(f"\n{prefix} Data Sample:")
        print(self.df.head(sample_size))

        # Create directory if it doesn't exist
        os.makedirs('charts/dataViz', exist_ok=True)

        # Save the sample to a text file
        with open(f'charts/dataViz/{prefix.lower().replace(" ", "_")}_data_sample.txt', 'w') as f:
            f.write(f"{prefix} Data Sample\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Shape: {self.df.shape}\n\n")
            f.write("Data Types:\n")
            for col, dtype in self.df.dtypes.items():
                f.write(f"{col}: {dtype}\n")
            f.write("\n" + "=" * 50 + "\n\n")
            f.write(self.df.head(sample_size).to_string())

            # Add some statistics
            f.write("\n\n" + "=" * 50 + "\n\n")
            f.write("Numerical Statistics:\n")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                f.write(self.df[numeric_cols].describe().to_string())
            else:
                f.write("No numerical columns found.")

    def clean_data(self):
        """Clean the data by handling dates and converting strings to lists."""
        # Print and save the raw data sample
        self.print_and_save_sample("Before Processing")

        # Convert skills strings to lists
        for col in ['skills', 'desired_skills']:
            self.df[col] = self.df[col].apply(self._string_to_list)

        # Convert success to boolean
        self.df['success'] = self.df['success'].astype(bool)

        # Add additional derived features
        self.df['skill_count_current'] = self.df['current_skills'].apply(len)
        self.df['skill_count_desired'] = self.df['desired_skills'].apply(len)
        self.df['skill_count_target'] = self.df['target_skills'].apply(len)

        # Calculate skill gaps (target skills not in current)
        self.df['skill_gap'] = self.df.apply(
            lambda x: [skill for skill in x['target_skills'] if skill not in x['current_skills']],
            axis=1
        )
        self.df['skill_gap_count'] = self.df['skill_gap'].apply(len)

        # Calculate skill overlap between current and target
        self.df['skill_overlap'] = self.df.apply(
            lambda x: [skill for skill in x['target_skills'] if skill in x['current_skills']],
            axis=1
        )
        self.df['skill_overlap_count'] = self.df['skill_overlap'].apply(len)

        # Calculate learning efficiency (overlap count / target count)
        self.df['learning_efficiency'] = self.df.apply(
            lambda x: x['skill_overlap_count'] / len(x['target_skills']) if len(x['target_skills']) > 0 else 0,
            axis=1
        )

        # Print and save the processed data sample
        self.print_and_save_sample("After Processing")

        return self

    def _string_to_list(self, skills_str):
        """Convert a string of comma-separated skills to a list."""
        if pd.isna(skills_str) or skills_str == "Unknown":
            return []

        # Remove quotes and split by comma
        skills_str = str(skills_str).strip('"')
        skills_list = [skill.strip() for skill in skills_str.split(',')]

        return skills_list

    def get_skills_stats(self, visualize=True):
        """Get statistics about skills."""
        all_skills = set()
        for col in ['current_skills', 'desired_skills', 'target_skills']:
            for skills_list in self.df[col]:
                all_skills.update(skills_list)

        stats = {
            'total_unique_skills': len(all_skills),
            'most_common_current': self._most_common_skills('current_skills'),
            'most_common_desired': self._most_common_skills('desired_skills'),
            'most_common_target': self._most_common_skills('target_skills')
        }

        if visualize:
            self._visualize_skills_stats()

        return stats

    def _most_common_skills(self, col, top_n=5):
        """Get the most common skills in a column."""
        all_skills = []
        for skills_list in self.df[col]:
            all_skills.extend(skills_list)

        skill_counts = pd.Series(all_skills).value_counts().head(top_n)
        return skill_counts.to_dict()

    def _visualize_skills_stats(self):
        """Create visualizations of skills statistics."""
        # Plot most common skills
        plt.figure(figsize=(18, 15))

        # Create subplots for each skill type
        skill_types = ['current_skills', 'desired_skills', 'target_skills']
        titles = ['Most Common Current Skills',
                  'Most Common Desired Skills',
                  'Most Common Target Skills']

        for i, (skill_type, title) in enumerate(zip(skill_types, titles), 1):
            plt.subplot(3, 1, i)
            skills = []
            for skill_list in self.df[skill_type]:
                skills.extend(skill_list)

            top_skills = pd.Series(skills).value_counts().head(10)
            sns.barplot(x=top_skills.values, y=top_skills.index, palette='viridis')
            plt.title(title, fontsize=14)
            plt.xlabel('Count', fontsize=12)
            plt.tight_layout()

        plt.savefig('charts/dataViz/common_skills.png')
        plt.close()

        # Create a network graph of skills
        self._create_skills_network()

    def _create_skills_network(self):
        """Create a network graph showing relationships between skills."""
        # Create a graph
        G = nx.Graph()

        # Add nodes for each skill
        all_skills = set()
        for col in ['current_skills', 'desired_skills', 'target_skills']:
            for skills_list in self.df[col]:
                all_skills.update(skills_list)

        for skill in all_skills:
            G.add_node(skill)

        # Add edges between skills that appear together
        for _, row in self.df.iterrows():
            current = set(row['current_skills'])
            target = set(row['target_skills'])

            # Connect skills that are learned together
            for s1 in current:
                for s2 in current:
                    if s1 != s2:
                        if G.has_edge(s1, s2):
                            G[s1][s2]['weight'] += 1
                        else:
                            G.add_edge(s1, s2, weight=1)

            # Connect skills that are targeted together
            for s1 in target:
                for s2 in target:
                    if s1 != s2:
                        if G.has_edge(s1, s2):
                            G[s1][s2]['weight'] += 1
                        else:
                            G.add_edge(s1, s2, weight=1)

        # Limit to top edges for visibility
        edges_with_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        edges_with_weights.sort(key=lambda x: x[2], reverse=True)
        top_edges = edges_with_weights[:50]

        H = nx.Graph()
        for u, v, w in top_edges:
            H.add_edge(u, v, weight=w)

        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(H, k=0.3)

        # Draw nodes
        nx.draw_networkx_nodes(H, pos, node_size=1000, alpha=0.7, node_color='skyblue')

        # Draw edges with width based on weight
        edge_widths = [H[u][v]['weight'] * 0.1 for u, v in H.edges()]
        nx.draw_networkx_edges(H, pos, width=edge_widths, alpha=0.4)

        # Draw labels
        nx.draw_networkx_labels(H, pos, font_size=10)

        plt.title('Skill Relationship Network', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('charts/dataViz/skill_network.png')
        plt.close()

    def success_rate_by_skill(self, skill_column='current_skills', visualize=True):
        """Calculate success rate per skill."""
        skill_success = {}

        for skill in set().union(*self.df[skill_column]):
            # Filter rows where the skill exists
            skill_df = self.df[[skill in skills for skills in self.df[skill_column]]]

            if len(skill_df) > 0:
                success_rate = skill_df['success'].mean() * 100
                skill_success[skill] = success_rate

        if visualize and skill_success:
            self._visualize_success_rates(skill_success)

        return skill_success

    def _visualize_success_rates(self, skill_success):
        """Visualize success rates by skill."""
        # Sort skills by success rate
        sorted_success = {k: v for k, v in sorted(skill_success.items(),
                                                  key=lambda item: item[1],
                                                  reverse=True)}

        # Take top 15 skills
        top_skills = dict(list(sorted_success.items())[:15])

        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(top_skills.values()), y=list(top_skills.keys()), palette='coolwarm')
        plt.title('Success Rate by Skill', fontsize=14)
        plt.xlabel('Success Rate (%)', fontsize=12)
        plt.axvline(x=50, color='gray', linestyle='--')
        plt.tight_layout()
        plt.savefig('charts/dataViz/success_rate_by_skill.png')
        plt.close()

    def skill_gap_analysis(self, visualize=True):
        """Analyze skill gaps and visualize them."""
        # Extract all gaps
        all_gaps = []
        for gaps in self.df['skill_gap']:
            all_gaps.extend(gaps)

        # Count occurrences of each gap
        gap_counts = pd.Series(all_gaps).value_counts()

        if visualize:
            plt.figure(figsize=(12, 8))
            top_gaps = gap_counts.head(10)
            sns.barplot(x=top_gaps.values, y=top_gaps.index, palette='magma')
            plt.title('Top 10 Skill Gaps', fontsize=14)
            plt.xlabel('Count', fontsize=12)
            plt.tight_layout()
            plt.savefig('charts/dataViz/top_skill_gaps.png')
            plt.close()

        return gap_counts.to_dict()

    def save_processed_data(self, output_path):
        """Save the processed dataframe to a CSV file."""
        try:
            # Convert lists back to strings for saving
            df_to_save = self.df.copy()
            for col in ['current_skills', 'desired_skills', 'target_skills', 'skill_gap', 'skill_overlap']:
                if col in df_to_save.columns:
                    df_to_save[col] = df_to_save[col].apply(lambda x: ', '.join(x))

            df_to_save.to_csv(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False