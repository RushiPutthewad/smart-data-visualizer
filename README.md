# Intelligent Data Visualizer

An automated Python tool that analyzes CSV data and generates intelligent, appropriate visualizations based on data types and relationships.

## Features

- **Automatic Data Type Detection**: Identifies numerical, categorical, and datetime columns
- **Intelligent Chart Selection**: Chooses appropriate visualizations based on data characteristics
- **Comprehensive Analysis**: Generates 3-5 relevant charts automatically
- **Data Quality Checks**: Identifies missing values, duplicates, and outliers
- **Statistical Summary**: Provides descriptive statistics and correlation analysis
- **Insights Report**: Generates actionable insights and recommendations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line
```bash
python intelligent_data_visualizer.py your_data.csv
```

### Python Script
```python
from intelligent_data_visualizer import IntelligentDataVisualizer

analyzer = IntelligentDataVisualizer('your_data.csv')
analyzer.run_analysis()
```

### Example with Sample Data
```bash
python example_usage.py
```

## Chart Types Generated

The tool automatically selects from these visualization types:

- **Distribution Analysis**: Histograms, box plots, density plots
- **Correlation Analysis**: Heatmaps, scatter plots
- **Categorical Analysis**: Bar charts, pie charts (≤6 categories), count plots
- **Comparison Analysis**: Box plots and violin plots by category
- **Time Series**: Line charts and area plots for temporal data

## Output Files

- **PNG Charts**: High-resolution visualizations (300 DPI)
- **analysis_insights.txt**: Key findings and recommendations
- **summary_statistics.csv**: Descriptive statistics table

## Error Prevention

- Validates data before plotting
- Handles missing values gracefully
- Prevents inappropriate chart types (e.g., pie charts with >6 categories)
- Type-safe operations with proper error handling

## Requirements

- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0