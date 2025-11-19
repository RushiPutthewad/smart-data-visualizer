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
    st.markdown("Upload a CSV file and explore different visualization types")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        visualizer = StreamlitVisualizer(df)
        
        # Sidebar navigation
        st.sidebar.header("📊 Dataset Info")
        st.sidebar.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.sidebar.write(f"Numerical: {len(visualizer.numerical_cols)}")
        st.sidebar.write(f"Categorical: {len(visualizer.categorical_cols)}")
        st.sidebar.write(f"DateTime: {len(visualizer.datetime_cols)}")
        
        st.sidebar.markdown("---")
        st.sidebar.header("🎨 Chart Pages")
        
        # Page selection
        pages = ["Data Overview", "Distribution Charts", "Correlation Analysis", 
                "Categorical Charts", "Bar Charts", "Line Charts", "Pie Charts", "Scatter Charts", "Histogram Charts", "Relationship Analysis", "Comparison Charts"]
        selected_page = st.sidebar.selectbox("Select Chart Type", pages)
        
        # Page routing
        if selected_page == "Data Overview":
            show_data_overview(df, visualizer)
        elif selected_page == "Distribution Charts":
            show_distribution_charts(df, visualizer)
        elif selected_page == "Correlation Analysis":
            show_correlation_analysis(df, visualizer)
        elif selected_page == "Categorical Charts":
            show_categorical_charts(df, visualizer)
        elif selected_page == "Bar Charts":
            show_bar_charts(df, visualizer)
        elif selected_page == "Line Charts":
            show_line_charts(df, visualizer)
        elif selected_page == "Pie Charts":
            show_pie_charts(df, visualizer)
        elif selected_page == "Scatter Charts":
            show_scatter_charts(df, visualizer)
        elif selected_page == "Histogram Charts":
            show_histogram_charts(df, visualizer)
        elif selected_page == "Relationship Analysis":
            show_relationship_analysis(df, visualizer)
        elif selected_page == "Comparison Charts":
            show_comparison_charts(df, visualizer)

def show_data_overview(df, visualizer):
    st.header("📊 Data Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Column Types")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': [str(df[col].dtype) for col in df.columns],
            'Non-Null': [df[col].count() for col in df.columns],
            'Unique': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info)
    
    if visualizer.numerical_cols:
        st.subheader("📈 Summary Statistics")
        st.dataframe(df[visualizer.numerical_cols].describe())
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.subheader("⚠️ Missing Values")
        st.bar_chart(missing[missing > 0])

def show_distribution_charts(df, visualizer):
    st.header("📈 Distribution Charts")
    
    if not visualizer.numerical_cols:
        st.warning("No numerical columns found for distribution analysis.")
        return
    
    selected_col = st.selectbox("Select Column for Distribution Analysis", visualizer.numerical_cols)
    
    if selected_col and not df[selected_col].dropna().empty:
        fig = visualizer.create_distribution_plot(selected_col)
        st.pyplot(fig)
        plt.close()
        
        # Show statistics
        col_data = df[selected_col].dropna()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{col_data.mean():.2f}")
        with col2:
            st.metric("Median", f"{col_data.median():.2f}")
        with col3:
            st.metric("Std Dev", f"{col_data.std():.2f}")
        with col4:
            st.metric("Skewness", f"{col_data.skew():.2f}")

def show_correlation_analysis(df, visualizer):
    st.header("🔗 Correlation Analysis")
    
    if len(visualizer.numerical_cols) < 2:
        st.warning("Need at least 2 numerical columns for correlation analysis.")
        return
    
    corr_fig = visualizer.create_correlation_heatmap()
    if corr_fig:
        st.pyplot(corr_fig)
        plt.close()
        
        # Show correlation table
        st.subheader("Correlation Matrix")
        corr_matrix = df[visualizer.numerical_cols].corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

def show_categorical_charts(df, visualizer):
    st.header("📊 Categorical Charts")
    
    if not visualizer.categorical_cols:
        st.warning("No categorical columns found.")
        return
    
    selected_col = st.selectbox("Select Categorical Column", visualizer.categorical_cols)
    
    if selected_col and not df[selected_col].dropna().empty:
        fig = visualizer.create_categorical_plot(selected_col)
        st.pyplot(fig)
        plt.close()
        
        # Show value counts
        st.subheader("Value Counts")
        value_counts = df[selected_col].value_counts()
        st.dataframe(value_counts.to_frame('Count'))

def show_bar_charts(df, visualizer):
    st.header("📊 Bar Charts")
    
    if not visualizer.categorical_cols:
        st.warning("No categorical columns found for bar charts.")
        return
    
    # Select columns and chart type
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_col = st.selectbox("Select Category Column", visualizer.categorical_cols)
    
    with col2:
        chart_type = st.selectbox("Chart Type", ["Counts", "Sum/Total", "Average", "Min/Max", "Percentages"])
    
    with col3:
        orientation = st.selectbox("Orientation", ["Vertical", "Horizontal"])
    
    # Show numerical column selector for non-count charts
    if chart_type != "Counts" and chart_type != "Percentages":
        if not visualizer.numerical_cols:
            st.warning("Need numerical columns for this chart type.")
            return
        num_col = st.selectbox("Select Value Column", visualizer.numerical_cols)
    
    if cat_col:
        # Calculate values based on chart type
        if chart_type == "Counts":
            values = df[cat_col].value_counts().head(15)
            y_label = "Count"
        elif chart_type == "Percentages":
            counts = df[cat_col].value_counts().head(15)
            values = (counts / counts.sum() * 100)
            y_label = "Percentage (%)"
        elif chart_type == "Sum/Total":
            values = df.groupby(cat_col)[num_col].sum().head(15)
            y_label = f"Sum of {num_col}"
        elif chart_type == "Average":
            values = df.groupby(cat_col)[num_col].mean().head(15)
            y_label = f"Average {num_col}"
        elif chart_type == "Min/Max":
            min_vals = df.groupby(cat_col)[num_col].min().head(15)
            max_vals = df.groupby(cat_col)[num_col].max().head(15)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            if orientation == "Vertical":
                ax1.bar(range(len(min_vals)), min_vals.values, color='lightcoral')
                ax1.set_xticks(range(len(min_vals)))
                ax1.set_xticklabels(min_vals.index, rotation=45, ha='right')
                ax1.set_ylabel(f"Min {num_col}")
                ax1.set_title(f"Minimum {num_col} by {cat_col}")
                
                ax2.bar(range(len(max_vals)), max_vals.values, color='lightblue')
                ax2.set_xticks(range(len(max_vals)))
                ax2.set_xticklabels(max_vals.index, rotation=45, ha='right')
                ax2.set_ylabel(f"Max {num_col}")
                ax2.set_title(f"Maximum {num_col} by {cat_col}")
            else:
                ax1.barh(range(len(min_vals)), min_vals.values, color='lightcoral')
                ax1.set_yticks(range(len(min_vals)))
                ax1.set_yticklabels(min_vals.index)
                ax1.set_xlabel(f"Min {num_col}")
                ax1.set_title(f"Minimum {num_col} by {cat_col}")
                
                ax2.barh(range(len(max_vals)), max_vals.values, color='lightblue')
                ax2.set_yticks(range(len(max_vals)))
                ax2.set_yticklabels(max_vals.index)
                ax2.set_xlabel(f"Max {num_col}")
                ax2.set_title(f"Maximum {num_col} by {cat_col}")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show min/max table
            st.subheader("Min/Max Values")
            minmax_df = pd.DataFrame({
                'Category': min_vals.index,
                'Minimum': min_vals.values,
                'Maximum': max_vals.values
            })
            st.dataframe(minmax_df)
            return
        
        # Create single bar chart for other types
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if orientation == "Vertical":
            ax.bar(range(len(values)), values.values, color='steelblue')
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=45, ha='right')
            ax.set_ylabel(y_label)
        else:
            ax.barh(range(len(values)), values.values, color='steelblue')
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(values.index)
            ax.set_xlabel(y_label)
        
        ax.set_title(f'{chart_type} of {cat_col}')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show data table
        st.subheader("Data Table")
        st.dataframe(values.to_frame(y_label))

def show_line_charts(df, visualizer):
    st.header("📈 Line Charts")
    
    if not visualizer.numerical_cols and not visualizer.datetime_cols:
        st.warning("Need numerical or datetime columns for line charts.")
        return
    
    # Column selection
    col1, col2 = st.columns(2)
    with col1:
        y_col = st.selectbox("Select Y-axis (Values)", visualizer.numerical_cols)
    
    with col2:
        x_options = ["Index"] + visualizer.datetime_cols + visualizer.numerical_cols
        x_col = st.selectbox("Select X-axis", x_options)
    
    if y_col:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if x_col == "Index":
            ax.plot(df.index, df[y_col], marker='o', linewidth=2, markersize=4)
            ax.set_xlabel('Index')
        else:
            # Sort by x_col for proper line chart
            sorted_df = df.sort_values(x_col)
            ax.plot(sorted_df[x_col], sorted_df[y_col], marker='o', linewidth=2, markersize=4)
            ax.set_xlabel(x_col)
            if x_col in visualizer.datetime_cols:
                plt.xticks(rotation=45)
        
        ax.set_ylabel(y_col)
        ax.set_title(f'Line Chart: {y_col} over {x_col}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show basic statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min", f"{df[y_col].min():.2f}")
        with col2:
            st.metric("Max", f"{df[y_col].max():.2f}")
        with col3:
            st.metric("Range", f"{df[y_col].max() - df[y_col].min():.2f}")

def show_pie_charts(df, visualizer):
    st.header("🍰 Pie Charts")
    
    if not visualizer.categorical_cols:
        st.warning("No categorical columns found for pie charts.")
        return
    
    selected_col = st.selectbox("Select Categorical Column", visualizer.categorical_cols)
    
    if selected_col and not df[selected_col].dropna().empty:
        value_counts = df[selected_col].value_counts()
        
        if len(value_counts) > 10:
            st.warning(f"Too many categories ({len(value_counts)}) for pie chart. Showing top 10.")
            value_counts = value_counts.head(10)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(range(len(value_counts)))
        
        wedges, texts, autotexts = ax.pie(value_counts.values, 
                                         labels=value_counts.index, 
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90)
        
        ax.set_title(f'Pie Chart of {selected_col}', fontsize=16, pad=20)
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show value counts and percentages
        st.subheader("Distribution Details")
        total = value_counts.sum()
        details_df = pd.DataFrame({
            'Category': value_counts.index,
            'Count': value_counts.values,
            'Percentage': [f"{(count/total)*100:.1f}%" for count in value_counts.values]
        })
        st.dataframe(details_df)

def show_scatter_charts(df, visualizer):
    st.header("🔴 Scatter Charts")
    
    if len(visualizer.numerical_cols) < 2:
        st.warning("Need at least 2 numerical columns for scatter charts.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select X-axis", visualizer.numerical_cols)
    with col2:
        y_col = st.selectbox("Select Y-axis", [col for col in visualizer.numerical_cols if col != x_col])
    
    if x_col and y_col:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50, color='steelblue')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
        ax.grid(True, alpha=0.3)
        
        # Calculate and display correlation
        corr = df[x_col].corr(df[y_col])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
                fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show correlation strength interpretation
        st.subheader("Correlation Analysis")
        if abs(corr) >= 0.7:
            strength = "Strong"
        elif abs(corr) >= 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        direction = "Positive" if corr > 0 else "Negative"
        st.write(f"**{strength} {direction} Correlation** ({corr:.3f})")

def show_histogram_charts(df, visualizer):
    st.header("📉 Histogram Charts")
    
    if not visualizer.numerical_cols:
        st.warning("No numerical columns found for histogram charts.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox("Select Column", visualizer.numerical_cols)
    with col2:
        bins = st.slider("Number of Bins", min_value=5, max_value=100, value=30)
    
    if selected_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = df[selected_col].dropna()
        
        ax.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel(selected_col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of {selected_col}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show statistics
        st.subheader("Distribution Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{data.mean():.2f}")
        with col2:
            st.metric("Std Dev", f"{data.std():.2f}")
        with col3:
            st.metric("Skewness", f"{data.skew():.2f}")
        with col4:
            st.metric("Kurtosis", f"{data.kurtosis():.2f}")

def show_relationship_analysis(df, visualizer):
    st.header("🔍 Relationship Analysis")
    
    if len(visualizer.numerical_cols) < 2:
        st.warning("Need at least 2 numerical columns for relationship analysis.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select X-axis", visualizer.numerical_cols)
    with col2:
        y_col = st.selectbox("Select Y-axis", [col for col in visualizer.numerical_cols if col != x_col])
    
    if x_col and y_col:
        fig = visualizer.create_scatter_plot(x_col, y_col)
        st.pyplot(fig)
        plt.close()

def show_comparison_charts(df, visualizer):
    st.header("⚖️ Comparison Charts")
    
    if not visualizer.numerical_cols or not visualizer.categorical_cols:
        st.warning("Need both numerical and categorical columns for comparison analysis.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        num_col = st.selectbox("Select Numerical Column", visualizer.numerical_cols)
    with col2:
        cat_col = st.selectbox("Select Categorical Column", visualizer.categorical_cols)
    
    if num_col and cat_col:
        fig = visualizer.create_comparison_plot(num_col, cat_col)
        if fig:
            st.pyplot(fig)
            plt.close()
        else:
            st.warning(f"Too many categories in {cat_col} (>10). Please select a column with fewer categories.")

if __name__ == "__main__":
    main()