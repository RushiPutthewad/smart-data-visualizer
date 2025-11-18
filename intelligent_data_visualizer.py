import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class IntelligentDataVisualizer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.insights = []
        
        # Set seaborn style
        sns.set_theme(style="whitegrid")
        plt.style.use('seaborn-v0_8-colorblind')
    
    def load_and_analyze_data(self):
        """Load CSV and perform initial analysis"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✓ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Basic info
            print(f"\nDataset Info:")
            print(f"Shape: {self.df.shape}")
            print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Missing values
            missing = self.df.isnull().sum()
            if missing.sum() > 0:
                print(f"\nMissing Values:")
                print(missing[missing > 0])
            
            # Data quality checks
            duplicates = self.df.duplicated().sum()
            if duplicates > 0:
                print(f"\nDuplicate rows: {duplicates}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def detect_column_types(self):
        """Automatically detect and categorize column types"""
        for col in self.df.columns:
            # Try datetime conversion
            if self.df[col].dtype == 'object':
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(self.df[col].dropna().head(100))
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        self.datetime_cols.append(col)
                        continue
                except:
                    pass
            
            # Numerical columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                unique_vals = self.df[col].nunique()
                if unique_vals > 2 and unique_vals > len(self.df) * 0.05:
                    self.numerical_cols.append(col)
                else:
                    self.categorical_cols.append(col)
            else:
                self.categorical_cols.append(col)
        
        print(f"\nColumn Classification:")
        print(f"Numerical: {self.numerical_cols}")
        print(f"Categorical: {self.categorical_cols}")
        print(f"Datetime: {self.datetime_cols}")
    
    def generate_summary_statistics(self):
        """Generate and display summary statistics"""
        if self.numerical_cols:
            print(f"\nNumerical Columns Statistics:")
            stats = self.df[self.numerical_cols].describe()
            print(stats)
            
            # Save statistics
            stats.to_csv('summary_statistics.csv')
    
    def create_single_numerical_plots(self):
        """Create plots for individual numerical columns"""
        charts_created = 0
        
        for col in self.numerical_cols[:3]:  # Limit to first 3 columns
            if self.df[col].dropna().empty:
                continue
                
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Histogram
            axes[0].hist(self.df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].set_title(f'Distribution of {col}')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Frequency')
            
            # Box plot
            axes[1].boxplot(self.df[col].dropna())
            axes[1].set_title(f'Box Plot of {col}')
            axes[1].set_ylabel(col)
            
            # Density plot
            self.df[col].dropna().plot.density(ax=axes[2], color='orange')
            axes[2].set_title(f'Density Plot of {col}')
            axes[2].set_xlabel(col)
            
            plt.tight_layout()
            plt.savefig(f'distribution_{col.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            charts_created += 1
            self.insights.append(f"Distribution analysis for {col}: Mean={self.df[col].mean():.2f}, Std={self.df[col].std():.2f}")
        
        return charts_created
    
    def create_correlation_analysis(self):
        """Create correlation heatmap for numerical columns"""
        if len(self.numerical_cols) < 2:
            return 0
            
        corr_matrix = self.df[self.numerical_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.2f}")
        
        if strong_corr:
            self.insights.append(f"Strong correlations found: {'; '.join(strong_corr)}")
        
        return 1
    
    def create_categorical_analysis(self):
        """Create plots for categorical columns"""
        charts_created = 0
        
        for col in self.categorical_cols[:3]:  # Process up to 3 categorical columns
            try:
                # STEP 1: Get value counts (like the example)
                category_counts = self.df[col].value_counts()
                
                # Print category counts for debugging
                print(f"Category Counts for {col}:")
                print(category_counts)
                
                # STEP 2: Validate data
                if category_counts.empty:
                    print(f"Skipped {col}: No data")
                    continue
                
                unique_categories = len(category_counts)
                total_count = category_counts.sum()
                max_category_pct = (category_counts.iloc[0] / total_count) * 100
                
                # Validation criteria
                valid_data = (
                    unique_categories >= 2 and
                    total_count > 10 and
                    max_category_pct < 95 and
                    2 <= unique_categories <= 6
                )
                
                # STEP 3: Generate appropriate chart based on validation
                if valid_data:
                    # Generate pie chart AND count plot side-by-side
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Pie chart using exact pattern from example
                    ax1.pie(category_counts.values, 
                            labels=category_counts.index, 
                            autopct='%1.1f%%',
                            startangle=90)
                    ax1.set_title(f'{col} Distribution')
                    
                    # Count plot
                    sns.countplot(data=self.df, x=col, ax=ax2)
                    ax2.set_title(f'Count Plot of {col}')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    # STEP 4: Save and log
                    plt.tight_layout()
                    plt.savefig(f'pie_chart_{col.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Pie chart created for {col}: {unique_categories} categories")
                    self.insights.append(f"Pie chart created for {col}: {unique_categories} categories")
                    
                elif unique_categories > 6:
                    # Generate only horizontal bar chart (top 10 categories)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_categories = category_counts.head(10)
                    sns.barplot(x=top_categories.values, y=top_categories.index, ax=ax)
                    ax.set_title(f'Top 10 Categories in {col}')
                    ax.set_xlabel('Count')
                    
                    plt.tight_layout()
                    plt.savefig(f'bar_chart_{col.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Bar chart created for {col}: {unique_categories} categories (too many for pie)")
                    self.insights.append(f"Bar chart created for {col}: {unique_categories} categories (too many for pie)")
                    
                else:
                    # Skip chart generation, log reason
                    reasons = []
                    if unique_categories < 2:
                        reasons.append("insufficient categories")
                    if total_count <= 10:
                        reasons.append("insufficient data")
                    if max_category_pct >= 95:
                        reasons.append("single category dominates")
                    
                    reason = ", ".join(reasons) if reasons else "validation failed"
                    print(f"Pie chart skipped for {col}: {reason}")
                    self.insights.append(f"Pie chart skipped for {col}: {reason}")
                    continue
                
                charts_created += 1
                
            except Exception as e:
                print(f"Error processing {col}: {e}")
                continue
        
        return charts_created
    
    def create_numerical_vs_categorical(self):
        """Create plots comparing numerical and categorical columns"""
        if not self.numerical_cols or not self.categorical_cols:
            return 0
            
        charts_created = 0
        
        for num_col in self.numerical_cols[:2]:
            for cat_col in self.categorical_cols[:1]:
                if self.df[cat_col].nunique() > 10:
                    continue
                    
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Box plot
                sns.boxplot(data=self.df, x=cat_col, y=num_col, ax=ax1)
                ax1.set_title(f'{num_col} by {cat_col}')
                ax1.tick_params(axis='x', rotation=45)
                
                # Violin plot
                sns.violinplot(data=self.df, x=cat_col, y=num_col, ax=ax2)
                ax2.set_title(f'{num_col} Distribution by {cat_col}')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(f'comparison_{num_col.replace(" ", "_")}_vs_{cat_col.replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                charts_created += 1
                self.insights.append(f"Comparison analysis: {num_col} varies across {cat_col} categories")
                
                if charts_created >= 2:  # Limit comparisons
                    break
            if charts_created >= 2:
                break
        
        return charts_created
    
    def create_time_series_analysis(self):
        """Create time series plots if datetime columns exist"""
        if not self.datetime_cols or not self.numerical_cols:
            return 0
            
        charts_created = 0
        
        for date_col in self.datetime_cols[:1]:
            for num_col in self.numerical_cols[:2]:
                # Sort by date
                df_sorted = self.df.sort_values(date_col)
                
                plt.figure(figsize=(12, 6))
                plt.plot(df_sorted[date_col], df_sorted[num_col], marker='o', markersize=3)
                plt.title(f'{num_col} over Time ({date_col})')
                plt.xlabel(date_col)
                plt.ylabel(num_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'timeseries_{num_col.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                charts_created += 1
                self.insights.append(f"Time series analysis: {num_col} trends over {date_col}")
        
        return charts_created
    
    def create_scatter_plots(self):
        """Create scatter plots for numerical column pairs"""
        if len(self.numerical_cols) < 2:
            return 0
            
        charts_created = 0
        
        # Create scatter plot for first two numerical columns
        for i in range(min(2, len(self.numerical_cols))):
            for j in range(i+1, min(i+3, len(self.numerical_cols))):
                col1, col2 = self.numerical_cols[i], self.numerical_cols[j]
                
                plt.figure(figsize=(8, 6))
                plt.scatter(self.df[col1], self.df[col2], alpha=0.6)
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.title(f'Scatter Plot: {col1} vs {col2}')
                
                # Add correlation coefficient
                corr = self.df[col1].corr(self.df[col2])
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                
                plt.tight_layout()
                plt.savefig(f'scatter_{col1.replace(" ", "_")}_vs_{col2.replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                charts_created += 1
                self.insights.append(f"Scatter analysis: {col1} vs {col2} correlation = {corr:.3f}")
                
                if charts_created >= 2:
                    break
            if charts_created >= 2:
                break
        
        return charts_created
    
    def save_insights_report(self):
        """Save analysis insights to a text file"""
        with open('analysis_insights.txt', 'w') as f:
            f.write("DATA ANALYSIS INSIGHTS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {self.csv_file}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"- Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n")
            f.write(f"- Numerical columns: {len(self.numerical_cols)}\n")
            f.write(f"- Categorical columns: {len(self.categorical_cols)}\n")
            f.write(f"- Datetime columns: {len(self.datetime_cols)}\n")
            f.write(f"- Missing values: {self.df.isnull().sum().sum()}\n\n")
            
            f.write("KEY INSIGHTS:\n")
            for i, insight in enumerate(self.insights, 1):
                f.write(f"{i}. {insight}\n")
            
            f.write("\nRECOMMendations:\n")
            if len(self.numerical_cols) > 1:
                f.write("- Consider feature engineering based on correlation patterns\n")
            if self.df.isnull().sum().sum() > 0:
                f.write("- Address missing values before modeling\n")
            if len(self.categorical_cols) > 0:
                f.write("- Consider encoding categorical variables for machine learning\n")
    
    def run_analysis(self):
        """Main method to run complete analysis"""
        print("🚀 Starting Intelligent Data Analysis...")
        
        if not self.load_and_analyze_data():
            return
        
        self.detect_column_types()
        self.generate_summary_statistics()
        
        print(f"\n📊 Generating visualizations...")
        total_charts = 0
        
        # Generate different types of visualizations
        total_charts += self.create_single_numerical_plots()
        total_charts += self.create_correlation_analysis()
        total_charts += self.create_categorical_analysis()
        total_charts += self.create_numerical_vs_categorical()
        total_charts += self.create_time_series_analysis()
        total_charts += self.create_scatter_plots()
        
        self.save_insights_report()
        
        print(f"\n✅ Analysis complete!")
        print(f"📈 Generated {total_charts} visualizations")
        print(f"📄 Saved insights to 'analysis_insights.txt'")
        print(f"📊 Saved statistics to 'summary_statistics.csv'")

