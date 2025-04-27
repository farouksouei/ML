import pandas as pd
import numpy as np
from DataProcessor import DataProcessor


def main():
    print("Starting DataProcessor test...")

    # Test with sample data
    print("\n1. Testing with sample data...")
    processor = DataProcessor(file_path='./data/edited_skill_exchange_dataset.csv', columns=8)
    processor.clean_data()

    # Test statistics and visualizations
    print("\n2. Getting skills statistics...")
    stats = processor.get_skills_stats()

    print("\n3. Analyzing success rates by skill...")
    success_rates = processor.success_rate_by_skill()
    for skill, rate in list(success_rates.items())[:5]:
        print(f"  - {skill}: {rate:.1f}%")

    print("\n4. Performing skill gap analysis...")
    gap_analysis = processor.skill_gap_analysis()
    for skill, count in list(gap_analysis.items())[:5]:
        print(f"  - {skill}: {count}")

    # Save processed data
    print("\n5. Saving processed data...")
    output_path = "charts/dataViz/processed_sample.csv"
    if processor.save_processed_data(output_path):
        print(f"Data successfully saved to {output_path}")
    else:
        print("Failed to save data")

    # Test with file loading (create a sample CSV first)
    print("\n6. Testing with file loading...")
    file_processor = DataProcessor(file_path='../data/edited_skill_exchange_dataset.csv', columns=6)
    file_processor.clean_data()

    print("\nTest completed!")


if __name__ == "__main__":
    main()