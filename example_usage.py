"""
Example usage of the Intelligent Data Visualizer
This script demonstrates how to use the visualizer with sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from intelligent_data_visualizer import IntelligentDataVisualizer

def create_sample_data():
    """Create sample dataset for demonstration"""
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    
    data = {
        'sales': np.random.normal(50000, 15000, n_samples),
        'marketing_spend': np.random.normal(5000, 1500, n_samples),
        'customer_satisfaction': np.random.uniform(1, 10, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
        'date': [datetime.now() - timedelta(days=x) for x in range(n_samples)],
        'is_weekend': np.random.choice([True, False], n_samples)
    }
    
    # Add some correlation
    data['profit'] = data['sales'] * 0.3 + np.random.normal(0, 2000, n_samples)
    
    df = pd.DataFrame(data)
    df.to_csv('sample_data.csv', index=False)
    print("✓ Sample data created: sample_data.csv")
    return 'sample_data.csv'

def main():
    """Run example analysis"""
    print("🔧 Creating sample dataset...")
    csv_file = create_sample_data()
    
    print("\n🚀 Running intelligent analysis...")
    analyzer = IntelligentDataVisualizer(csv_file)
    analyzer.run_analysis()
    
    print("\n📁 Generated files:")
    print("- Multiple PNG visualization files")
    print("- analysis_insights.txt")
    print("- summary_statistics.csv")

if __name__ == "__main__":
    main()