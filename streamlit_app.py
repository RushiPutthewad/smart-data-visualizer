import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime

# Set page config
st.set_page_config(page_title="Intelligent Data Visualizer", layout="wide")

# Set style
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8-colorblind')

class StreamlitVisualizer:
    def __init__(self, df):
        self.df = df
        self.numerical_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.detect_column_types()
    
    def detect_column_types(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col].dropna().head(100))
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    self.datetime_cols.append(col)
                    continue
                except:
                    pass
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                unique_vals = self.df[col].nunique()
                if unique_vals > 2 and unique_vals > len(self.df) * 0.05:
                    self.numerical_cols.append(col)
                else:
                    self.categorical_cols.append(col)
            else:
                self.categorical_cols.append(col)
    
    def create_distribution_plot(self, col):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(self.df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title(f'Distribution of {col}')
        axes[0].set_xlabel(col)
        
        axes[1].boxplot(self.df[col].dropna())
        axes[1].set_title(f'Box Plot of {col}')
        axes[1].set_ylabel(col)
        
        try:
            self.df[col].dropna().plot.density(ax=axes[2], color='orange')
            axes[2].set_title(f'Density Plot of {col}')
        except ImportError:
            # Fallback to histogram if scipy not available
            axes[2].hist(self.df[col].dropna(), bins=50, alpha=0.7, color='orange', density=True)
            axes[2].set_title(f'Normalized Distribution of {col}')
        
        plt.tight_layout()
        return fig
    
    def create_correlation_heatmap(self):
        if len(self.numerical_cols) < 2:
            return None
        
        corr_matrix = self.df[self.numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix')
        return fig
    
    def create_categorical_plot(self, col):
        value_counts = self.df[col].value_counts()
        
        if len(value_counts) <= 6:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            ax1.set_title(f'Distribution of {col}')
            sns.countplot(data=self.df, x=col, ax=ax2)
            ax2.set_title(f'Count Plot of {col}')
            ax2.tick_params(axis='x', rotation=45)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_categories = value_counts.head(10)
            sns.barplot(x=top_categories.values, y=top_categories.index, ax=ax)
            ax.set_title(f'Top 10 Categories in {col}')
        
        plt.tight_layout()
        return fig
    
    def create_scatter_plot(self, col1, col2):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.df[col1], self.df[col2], alpha=0.6)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f'Scatter Plot: {col1} vs {col2}')
        
        corr = self.df[col1].corr(self.df[col2])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        return fig
    
    def create_comparison_plot(self, num_col, cat_col):
        if self.df[cat_col].nunique() > 10:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, ax=ax1)
        ax1.set_title(f'{num_col} by {cat_col}')
        ax1.tick_params(axis='x', rotation=45)
        
        sns.violinplot(data=self.df, x=cat_col, y=num_col, ax=ax2)
        ax2.set_title(f'{num_col} Distribution by {cat_col}')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

def main():
    st.title("🎯 Intelligent Data Visualizer")
    st.markdown("Upload a CSV file and get automatic intelligent visualizations")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        visualizer = StreamlitVisualizer(df)
        
        # Sidebar info
        st.sidebar.header("Dataset Info")
        st.sidebar.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.sidebar.write(f"Numerical: {len(visualizer.numerical_cols)}")
        st.sidebar.write(f"Categorical: {len(visualizer.categorical_cols)}")
        st.sidebar.write(f"DateTime: {len(visualizer.datetime_cols)}")
        
        # Data preview
        st.header("📊 Data Preview")
        st.dataframe(df.head())
        
        # Summary statistics
        if visualizer.numerical_cols:
            st.header("📈 Summary Statistics")
            st.dataframe(df[visualizer.numerical_cols].describe())
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.header("⚠️ Missing Values")
            st.bar_chart(missing[missing > 0])
        
        # Visualizations
        st.header("🎨 Automatic Visualizations")
        
        # Distribution plots for numerical columns
        if visualizer.numerical_cols:
            st.subheader("Distribution Analysis")
            for col in visualizer.numerical_cols[:3]:
                if not df[col].dropna().empty:
                    fig = visualizer.create_distribution_plot(col)
                    st.pyplot(fig)
                    plt.close()
        
        # Correlation heatmap
        corr_fig = visualizer.create_correlation_heatmap()
        if corr_fig:
            st.subheader("Correlation Analysis")
            st.pyplot(corr_fig)
            plt.close()
        
        # Categorical analysis
        if visualizer.categorical_cols:
            st.subheader("Categorical Analysis")
            for col in visualizer.categorical_cols[:2]:
                if not df[col].dropna().empty:
                    fig = visualizer.create_categorical_plot(col)
                    st.pyplot(fig)
                    plt.close()
        
        # Scatter plots
        if len(visualizer.numerical_cols) >= 2:
            st.subheader("Relationship Analysis")
            col1, col2 = visualizer.numerical_cols[0], visualizer.numerical_cols[1]
            fig = visualizer.create_scatter_plot(col1, col2)
            st.pyplot(fig)
            plt.close()
        
        # Comparison plots
        if visualizer.numerical_cols and visualizer.categorical_cols:
            st.subheader("Comparison Analysis")
            num_col = visualizer.numerical_cols[0]
            cat_col = visualizer.categorical_cols[0]
            fig = visualizer.create_comparison_plot(num_col, cat_col)
            if fig:
                st.pyplot(fig)
                plt.close()
        
        # Download insights
        insights = []
        insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
        insights.append(f"Found {len(visualizer.numerical_cols)} numerical and {len(visualizer.categorical_cols)} categorical columns")
        if missing.sum() > 0:
            insights.append(f"Dataset has {missing.sum()} missing values")
        
        st.header("💡 Key Insights")
        for insight in insights:
            st.write(f"• {insight}")
        
        # Chart summary table
        st.header("📋 Chart Summary")
        chart_data = []
        
        # Add numerical columns
        for col in visualizer.numerical_cols[:3]:
            if not df[col].dropna().empty:
                chart_data.append({"Column Name": col, "Chart Type": "Distribution (Histogram, Box, Density)"})
        
        # Add correlation
        if len(visualizer.numerical_cols) >= 2:
            chart_data.append({"Column Name": "All Numerical", "Chart Type": "Correlation Heatmap"})
        
        # Add categorical columns
        for col in visualizer.categorical_cols[:2]:
            if not df[col].dropna().empty:
                chart_type = "Pie Chart & Count Plot" if df[col].nunique() <= 6 else "Bar Chart (Top 10)"
                chart_data.append({"Column Name": col, "Chart Type": chart_type})
        
        # Add scatter plot
        if len(visualizer.numerical_cols) >= 2:
            col1, col2 = visualizer.numerical_cols[0], visualizer.numerical_cols[1]
            chart_data.append({"Column Name": f"{col1} vs {col2}", "Chart Type": "Scatter Plot"})
        
        # Add comparison plot
        if visualizer.numerical_cols and visualizer.categorical_cols:
            num_col = visualizer.numerical_cols[0]
            cat_col = visualizer.categorical_cols[0]
            if df[cat_col].nunique() <= 10:
                chart_data.append({"Column Name": f"{num_col} by {cat_col}", "Chart Type": "Box Plot & Violin Plot"})
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            st.dataframe(chart_df, use_container_width=True)

if __name__ == "__main__":
    main()