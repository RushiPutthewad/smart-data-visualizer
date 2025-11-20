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
                "Categorical Charts", "Bar Charts", "Line Charts", "Pie Charts", "Scatter Charts", "Histogram Charts", "Box Plot Charts", "Treemaps Charts", "Sunburst Charts", "Area Charts", "Waterfall Charts", "Bubble Charts", "Funnel Charts", "Grouped/Stacked Charts", "Polar Area Diagrams", "Streamgraphs", "Relationship Analysis", "Comparison Charts"]
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
        elif selected_page == "Box Plot Charts":
            show_box_plot_charts(df, visualizer)
        elif selected_page == "Treemaps Charts":
            show_treemaps_charts(df, visualizer)
        elif selected_page == "Sunburst Charts":
            show_sunburst_charts(df, visualizer)
        elif selected_page == "Area Charts":
            show_area_charts(df, visualizer)
        elif selected_page == "Waterfall Charts":
            show_waterfall_charts(df, visualizer)
        elif selected_page == "Bubble Charts":
            show_bubble_charts(df, visualizer)
        elif selected_page == "Funnel Charts":
            show_funnel_charts(df, visualizer)
        elif selected_page == "Grouped/Stacked Charts":
            show_grouped_stacked_charts(df, visualizer)
        elif selected_page == "Polar Area Diagrams":
            show_polar_area_diagrams(df, visualizer)
        elif selected_page == "Streamgraphs":
            show_streamgraphs(df, visualizer)
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

def show_box_plot_charts(df, visualizer):
    st.header("📦 Box Plot Charts")
    
    if not visualizer.numerical_cols:
        st.warning("No numerical columns found for box plots.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        num_col = st.selectbox("Select Numerical Column", visualizer.numerical_cols)
    
    with col2:
        if visualizer.categorical_cols:
            group_by = st.selectbox("Group By (Optional)", ["None"] + visualizer.categorical_cols)
        else:
            group_by = "None"
    
    if num_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if group_by == "None":
            ax.boxplot(df[num_col].dropna())
            ax.set_ylabel(num_col)
            ax.set_title(f'Box Plot of {num_col}')
            ax.set_xticklabels([num_col])
        else:
            if df[group_by].nunique() > 10:
                st.warning(f"Too many categories in {group_by} (>10). Showing top 10.")
                top_cats = df[group_by].value_counts().head(10).index
                plot_df = df[df[group_by].isin(top_cats)]
            else:
                plot_df = df
            
            sns.boxplot(data=plot_df, x=group_by, y=num_col, ax=ax)
            ax.set_title(f'Box Plot of {num_col} by {group_by}')
            ax.tick_params(axis='x', rotation=45)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show box plot statistics
        st.subheader("Box Plot Statistics")
        if group_by == "None":
            data = df[num_col].dropna()
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Q1 (25%)", f"{data.quantile(0.25):.2f}")
            with col2:
                st.metric("Median (50%)", f"{data.median():.2f}")
            with col3:
                st.metric("Q3 (75%)", f"{data.quantile(0.75):.2f}")
            with col4:
                st.metric("IQR", f"{data.quantile(0.75) - data.quantile(0.25):.2f}")
            with col5:
                outliers = len(data[(data < data.quantile(0.25) - 1.5*(data.quantile(0.75) - data.quantile(0.25))) | 
                                   (data > data.quantile(0.75) + 1.5*(data.quantile(0.75) - data.quantile(0.25)))])
                st.metric("Outliers", outliers)
        else:
            stats_df = df.groupby(group_by)[num_col].describe()
            st.dataframe(stats_df)

def show_treemaps_charts(df, visualizer):
    st.header("🌳 Treemaps Charts")
    
    if not visualizer.categorical_cols:
        st.warning("No categorical columns found for treemaps.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        cat_col = st.selectbox("Select Category Column", visualizer.categorical_cols)
    
    with col2:
        if visualizer.numerical_cols:
            value_type = st.selectbox("Value Type", ["Count", "Sum", "Average"])
            if value_type != "Count":
                num_col = st.selectbox("Select Value Column", visualizer.numerical_cols)
        else:
            value_type = "Count"
    
    if cat_col:
        # Calculate values based on type
        if value_type == "Count":
            values = df[cat_col].value_counts().head(20)
        elif value_type == "Sum":
            values = df.groupby(cat_col)[num_col].sum().head(20)
        elif value_type == "Average":
            values = df.groupby(cat_col)[num_col].mean().head(20)
        
        # Create treemap using matplotlib
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Simple treemap implementation
        total = values.sum()
        colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
        
        # Calculate rectangle sizes
        sizes = values.values / total
        
        # Simple grid layout for treemap
        x, y = 0, 0
        width, height = 1, 1
        
        for i, (label, size) in enumerate(zip(values.index, sizes)):
            if i < len(values) // 2:
                rect_width = size * 2
                rect_height = 0.5
                rect_x = x
                rect_y = y
                x += rect_width
            else:
                rect_width = size * 2
                rect_height = 0.5
                rect_x = x - rect_width
                rect_y = 0.5
                x -= rect_width
            
            # Create rectangle
            rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                   facecolor=colors[i], edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            
            # Add text if rectangle is large enough
            if rect_width > 0.1 and rect_height > 0.1:
                ax.text(rect_x + rect_width/2, rect_y + rect_height/2, 
                       f'{label}\n{values[label]:.1f}',
                       ha='center', va='center', fontsize=8, weight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Treemap: {value_type} by {cat_col}', fontsize=16, pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show data table
        st.subheader("Data Table")
        st.dataframe(values.to_frame(value_type))

def show_sunburst_charts(df, visualizer):
    st.header("☀️ Sunburst Charts")
    
    if len(visualizer.categorical_cols) < 2:
        st.warning("Need at least 2 categorical columns for sunburst charts.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        inner_col = st.selectbox("Select Inner Ring", visualizer.categorical_cols)
    with col2:
        outer_col = st.selectbox("Select Outer Ring", [col for col in visualizer.categorical_cols if col != inner_col])
    
    if inner_col and outer_col:
        # Create hierarchical data
        grouped = df.groupby([inner_col, outer_col]).size().reset_index(name='count')
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Inner ring data
        inner_counts = df[inner_col].value_counts().head(8)
        inner_total = inner_counts.sum()
        
        # Outer ring data
        outer_data = []
        for inner_cat in inner_counts.index:
            outer_subset = df[df[inner_col] == inner_cat][outer_col].value_counts().head(5)
            for outer_cat, count in outer_subset.items():
                outer_data.append((inner_cat, outer_cat, count))
        
        # Colors
        inner_colors = plt.cm.Set3(np.linspace(0, 1, len(inner_counts)))
        outer_colors = plt.cm.Pastel1(np.linspace(0, 1, len(outer_data)))
        
        # Draw inner ring
        theta_inner = np.linspace(0, 2*np.pi, len(inner_counts)+1)
        for i, (cat, count) in enumerate(inner_counts.items()):
            theta_start = theta_inner[i]
            theta_end = theta_inner[i+1]
            theta_mid = (theta_start + theta_end) / 2
            
            # Inner ring wedge
            theta_range = np.linspace(theta_start, theta_end, 50)
            r_inner = np.full_like(theta_range, 0.3)
            r_outer = np.full_like(theta_range, 0.6)
            
            ax.fill_between(theta_range, r_inner, r_outer, color=inner_colors[i], alpha=0.8)
            
            # Label
            if theta_end - theta_start > 0.2:  # Only label if segment is large enough
                ax.text(theta_mid, 0.45, str(cat)[:10], ha='center', va='center', 
                       rotation=np.degrees(theta_mid)-90 if theta_mid > np.pi/2 and theta_mid < 3*np.pi/2 else np.degrees(theta_mid)+90,
                       fontsize=8, weight='bold')
        
        # Draw outer ring
        current_theta = 0
        for inner_cat, outer_cat, count in outer_data:
            if inner_cat in inner_counts.index:
                # Calculate proportional angle
                inner_proportion = inner_counts[inner_cat] / inner_total
                segment_angle = 2 * np.pi * inner_proportion * (count / df[df[inner_col] == inner_cat].shape[0])
                
                theta_start = current_theta
                theta_end = current_theta + segment_angle
                theta_mid = (theta_start + theta_end) / 2
                
                # Outer ring wedge
                theta_range = np.linspace(theta_start, theta_end, 20)
                r_inner = np.full_like(theta_range, 0.6)
                r_outer = np.full_like(theta_range, 0.9)
                
                color_idx = list(inner_counts.index).index(inner_cat)
                ax.fill_between(theta_range, r_inner, r_outer, 
                               color=inner_colors[color_idx], alpha=0.5)
                
                # Label for larger segments
                if segment_angle > 0.1:
                    ax.text(theta_mid, 0.75, str(outer_cat)[:8], ha='center', va='center',
                           rotation=np.degrees(theta_mid)-90 if theta_mid > np.pi/2 and theta_mid < 3*np.pi/2 else np.degrees(theta_mid)+90,
                           fontsize=6)
                
                current_theta = theta_end
        
        ax.set_ylim(0, 1)
        ax.set_title(f'Sunburst Chart: {inner_col} (inner) & {outer_col} (outer)', 
                    fontsize=16, pad=30)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show hierarchy table
        st.subheader("Hierarchy Data")
        st.dataframe(grouped.head(20))

def show_area_charts(df, visualizer):
    st.header("🏔️ Area Charts")
    
    if not visualizer.numerical_cols:
        st.warning("No numerical columns found for area charts.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        y_col = st.selectbox("Select Y-axis (Values)", visualizer.numerical_cols)
    
    with col2:
        x_options = ["Index"] + visualizer.datetime_cols + visualizer.numerical_cols
        x_col = st.selectbox("Select X-axis", x_options)
    
    with col3:
        if visualizer.categorical_cols:
            stack_by = st.selectbox("Stack By (Optional)", ["None"] + visualizer.categorical_cols)
        else:
            stack_by = "None"
    
    if y_col:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if stack_by == "None":
            # Single area chart
            if x_col == "Index":
                ax.fill_between(df.index, 0, df[y_col], alpha=0.7, color='steelblue')
                ax.set_xlabel('Index')
            else:
                sorted_df = df.sort_values(x_col)
                ax.fill_between(sorted_df[x_col], 0, sorted_df[y_col], alpha=0.7, color='steelblue')
                ax.set_xlabel(x_col)
                if x_col in visualizer.datetime_cols:
                    plt.xticks(rotation=45)
            
            ax.set_ylabel(y_col)
            ax.set_title(f'Area Chart: {y_col} over {x_col}')
        
        else:
            # Stacked area chart
            if df[stack_by].nunique() > 8:
                st.warning(f"Too many categories in {stack_by} (>8). Showing top 8.")
                top_cats = df[stack_by].value_counts().head(8).index
                plot_df = df[df[stack_by].isin(top_cats)]
            else:
                plot_df = df
            
            # Pivot data for stacking
            if x_col == "Index":
                pivot_df = plot_df.pivot_table(index=plot_df.index, columns=stack_by, values=y_col, fill_value=0)
                x_data = pivot_df.index
            else:
                pivot_df = plot_df.pivot_table(index=x_col, columns=stack_by, values=y_col, fill_value=0)
                x_data = pivot_df.index
            
            # Create stacked area
            colors = plt.cm.Set3(np.linspace(0, 1, len(pivot_df.columns)))
            ax.stackplot(x_data, *[pivot_df[col] for col in pivot_df.columns], 
                        labels=pivot_df.columns, colors=colors, alpha=0.8)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Stacked Area Chart: {y_col} by {stack_by} over {x_col}')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            if x_col in visualizer.datetime_cols:
                plt.xticks(rotation=45)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show statistics
        st.subheader("Statistics")
        if stack_by == "None":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Area", f"{df[y_col].sum():.2f}")
            with col2:
                st.metric("Average", f"{df[y_col].mean():.2f}")
            with col3:
                st.metric("Peak Value", f"{df[y_col].max():.2f}")
        else:
            summary_df = df.groupby(stack_by)[y_col].agg(['sum', 'mean', 'max']).round(2)
            st.dataframe(summary_df)

def show_waterfall_charts(df, visualizer):
    st.header("🌊 Waterfall Charts")
    
    if not visualizer.numerical_cols or not visualizer.categorical_cols:
        st.warning("Need both numerical and categorical columns for waterfall charts.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        cat_col = st.selectbox("Select Category Column", visualizer.categorical_cols)
    with col2:
        num_col = st.selectbox("Select Value Column", visualizer.numerical_cols)
    
    if cat_col and num_col:
        # Get data and limit to top 10 categories
        data = df.groupby(cat_col)[num_col].sum().head(10)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate cumulative values
        cumulative = 0
        x_pos = range(len(data))
        
        # Colors for positive and negative values
        colors = ['green' if val >= 0 else 'red' for val in data.values]
        
        # Draw waterfall bars
        for i, (category, value) in enumerate(data.items()):
            # Draw connecting line from previous bar
            if i > 0:
                ax.plot([i-0.4, i-0.4], [cumulative, cumulative + value], 
                       color='gray', linestyle='--', alpha=0.5)
            
            # Draw the bar
            ax.bar(i, abs(value), bottom=cumulative if value >= 0 else cumulative + value,
                  color=colors[i], alpha=0.7, width=0.8)
            
            # Add value labels
            label_y = cumulative + value/2 if value >= 0 else cumulative + value + abs(value)/2
            ax.text(i, label_y, f'{value:.1f}', ha='center', va='center', 
                   fontweight='bold', color='white')
            
            cumulative += value
        
        # Draw final total line
        ax.axhline(y=cumulative, color='blue', linestyle='-', linewidth=2, alpha=0.7)
        ax.text(len(data)-1, cumulative + max(data.values)*0.05, f'Total: {cumulative:.1f}',
               ha='right', va='bottom', fontweight='bold', color='blue')
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(data.index, rotation=45, ha='right')
        ax.set_ylabel(num_col)
        ax.set_title(f'Waterfall Chart: {num_col} by {cat_col}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show data table
        st.subheader("Waterfall Data")
        waterfall_df = pd.DataFrame({
            'Category': data.index,
            'Value': data.values,
            'Cumulative': np.cumsum(data.values)
        })
        st.dataframe(waterfall_df)
        
        # Show summary
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", f"{cumulative:.2f}")
        with col2:
            positive_sum = data[data > 0].sum()
            st.metric("Positive Contributions", f"{positive_sum:.2f}")
        with col3:
            negative_sum = data[data < 0].sum()
            st.metric("Negative Contributions", f"{negative_sum:.2f}")

def show_bubble_charts(df, visualizer):
    st.header("🔵 Bubble Charts")
    
    if len(visualizer.numerical_cols) < 3:
        st.warning("Need at least 3 numerical columns for bubble charts.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x_col = st.selectbox("Select X-axis", visualizer.numerical_cols)
    with col2:
        y_col = st.selectbox("Select Y-axis", [col for col in visualizer.numerical_cols if col != x_col])
    with col3:
        size_col = st.selectbox("Select Bubble Size", [col for col in visualizer.numerical_cols if col not in [x_col, y_col]])
    with col4:
        if visualizer.categorical_cols:
            color_col = st.selectbox("Color By (Optional)", ["None"] + visualizer.categorical_cols)
        else:
            color_col = "None"
    
    if x_col and y_col and size_col:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize bubble sizes
        sizes = df[size_col].fillna(0)
        sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 1000 + 50
        
        if color_col == "None":
            scatter = ax.scatter(df[x_col], df[y_col], s=sizes, alpha=0.6, color='steelblue')
        else:
            # Color by categorical column
            if df[color_col].nunique() > 10:
                st.warning(f"Too many categories in {color_col} (>10). Showing top 10.")
                top_cats = df[color_col].value_counts().head(10).index
                plot_df = df[df[color_col].isin(top_cats)]
            else:
                plot_df = df
            
            categories = plot_df[color_col].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            
            for i, cat in enumerate(categories):
                mask = plot_df[color_col] == cat
                if mask.any():
                    cat_sizes = (plot_df.loc[mask, size_col].fillna(0) - sizes.min()) / (sizes.max() - sizes.min()) * 1000 + 50
                    ax.scatter(plot_df.loc[mask, x_col], plot_df.loc[mask, y_col], 
                             s=cat_sizes, alpha=0.6, color=colors[i], label=str(cat))
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Bubble Chart: {x_col} vs {y_col} (size: {size_col})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show correlations
        st.subheader("Correlations")
        col1, col2, col3 = st.columns(3)
        with col1:
            corr_xy = df[x_col].corr(df[y_col])
            st.metric(f"{x_col} vs {y_col}", f"{corr_xy:.3f}")
        with col2:
            corr_xs = df[x_col].corr(df[size_col])
            st.metric(f"{x_col} vs {size_col}", f"{corr_xs:.3f}")
        with col3:
            corr_ys = df[y_col].corr(df[size_col])
            st.metric(f"{y_col} vs {size_col}", f"{corr_ys:.3f}")
        
        # Show data sample
        st.subheader("Data Sample")
        sample_cols = [x_col, y_col, size_col]
        if color_col != "None":
            sample_cols.append(color_col)
        st.dataframe(df[sample_cols].head(10))

def show_funnel_charts(df, visualizer):
    st.header("📊 Funnel Charts")
    
    if not visualizer.categorical_cols or not visualizer.numerical_cols:
        st.warning("Need both categorical and numerical columns for funnel charts.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        cat_col = st.selectbox("Select Stage Column", visualizer.categorical_cols)
    with col2:
        num_col = st.selectbox("Select Value Column", visualizer.numerical_cols)
    
    if cat_col and num_col:
        # Get data and sort by values (descending for funnel)
        data = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(8)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate funnel widths
        max_val = data.max()
        widths = data.values / max_val
        
        # Colors from light to dark
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(data)))
        
        # Draw funnel segments
        y_pos = np.arange(len(data))
        for i, (stage, value) in enumerate(data.items()):
            width = widths[i]
            
            # Create trapezoid shape
            left = (1 - width) / 2
            right = left + width
            
            # Draw rectangle for each stage
            rect = plt.Rectangle((left, i - 0.4), width, 0.8, 
                               facecolor=colors[i], edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            
            # Add stage label
            ax.text(0.5, i, f'{stage}\n{value:.0f}', ha='center', va='center',
                   fontweight='bold', fontsize=10, color='white')
            
            # Add percentage if not first stage
            if i > 0:
                pct = (value / data.iloc[0]) * 100
                ax.text(right + 0.02, i, f'{pct:.1f}%', ha='left', va='center',
                       fontsize=9, color='gray')
        
        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(data) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f'Funnel Chart: {num_col} by {cat_col}', fontsize=16, pad=20)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show conversion rates
        st.subheader("Conversion Analysis")
        conversion_df = pd.DataFrame({
            'Stage': data.index,
            'Value': data.values,
            'Conversion Rate': [100.0] + [(data.iloc[i] / data.iloc[0] * 100) for i in range(1, len(data))],
            'Drop-off Rate': [0.0] + [((data.iloc[i-1] - data.iloc[i]) / data.iloc[i-1] * 100) for i in range(1, len(data))]
        })
        st.dataframe(conversion_df.round(2))
        
        # Show summary metrics
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stages", len(data))
        with col2:
            overall_conversion = (data.iloc[-1] / data.iloc[0] * 100) if len(data) > 1 else 100
            st.metric("Overall Conversion", f"{overall_conversion:.1f}%")
        with col3:
            total_dropoff = data.iloc[0] - data.iloc[-1] if len(data) > 1 else 0
            st.metric("Total Drop-off", f"{total_dropoff:.0f}")

def show_grouped_stacked_charts(df, visualizer):
    st.header("📊 Grouped/Stacked Charts")
    
    if len(visualizer.categorical_cols) < 2 or not visualizer.numerical_cols:
        st.warning("Need at least 2 categorical columns and 1 numerical column.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x_col = st.selectbox("Select X-axis (Groups)", visualizer.categorical_cols)
    with col2:
        stack_col = st.selectbox("Select Stack/Group By", [col for col in visualizer.categorical_cols if col != x_col])
    with col3:
        value_col = st.selectbox("Select Value Column", visualizer.numerical_cols)
    with col4:
        chart_style = st.selectbox("Chart Style", ["Grouped", "Stacked"])
    
    if x_col and stack_col and value_col:
        # Limit categories for readability
        if df[x_col].nunique() > 10:
            st.warning(f"Too many categories in {x_col} (>10). Showing top 10.")
            top_x = df[x_col].value_counts().head(10).index
            plot_df = df[df[x_col].isin(top_x)]
        else:
            plot_df = df
        
        if plot_df[stack_col].nunique() > 8:
            st.warning(f"Too many categories in {stack_col} (>8). Showing top 8.")
            top_stack = plot_df[stack_col].value_counts().head(8).index
            plot_df = plot_df[plot_df[stack_col].isin(top_stack)]
        
        # Pivot data
        pivot_df = plot_df.pivot_table(index=x_col, columns=stack_col, values=value_col, 
                                      aggfunc='sum', fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if chart_style == "Grouped":
            # Grouped bar chart
            pivot_df.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'Grouped Bar Chart: {value_col} by {x_col} and {stack_col}')
        else:
            # Stacked bar chart
            pivot_df.plot(kind='bar', stacked=True, ax=ax, width=0.8)
            ax.set_title(f'Stacked Bar Chart: {value_col} by {x_col} and {stack_col}')
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(value_col)
        ax.legend(title=stack_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show data table
        st.subheader("Pivot Table")
        st.dataframe(pivot_df)
        
        # Show summary statistics
        st.subheader("Summary by Groups")
        summary_df = pivot_df.agg(['sum', 'mean', 'max']).round(2)
        st.dataframe(summary_df)
        
        # Show totals
        st.subheader("Totals")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total by {x_col}:**")
            x_totals = pivot_df.sum(axis=1).sort_values(ascending=False)
            st.dataframe(x_totals.to_frame('Total'))
        with col2:
            st.write(f"**Total by {stack_col}:**")
            stack_totals = pivot_df.sum(axis=0).sort_values(ascending=False)
            st.dataframe(stack_totals.to_frame('Total'))

def show_polar_area_diagrams(df, visualizer):
    st.header("🌐 Polar Area Diagrams")
    
    if not visualizer.categorical_cols or not visualizer.numerical_cols:
        st.warning("Need both categorical and numerical columns for polar area diagrams.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        cat_col = st.selectbox("Select Category Column", visualizer.categorical_cols)
    with col2:
        num_col = st.selectbox("Select Value Column", visualizer.numerical_cols)
    
    if cat_col and num_col:
        # Get data and limit to top 12 categories
        data = df.groupby(cat_col)[num_col].sum().head(12)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each category
        angles = np.linspace(0, 2 * np.pi, len(data), endpoint=False)
        
        # Normalize values for radius (0.1 to 1.0)
        values = data.values
        max_val = values.max()
        radii = 0.1 + (values / max_val) * 0.9
        
        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        
        # Create polar area chart
        bars = ax.bar(angles, radii, width=2*np.pi/len(data), 
                     color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add category labels
        for angle, radius, category, value in zip(angles, radii, data.index, data.values):
            # Position labels outside the bars
            label_radius = radius + 0.15
            ax.text(angle, label_radius, f'{category}\n{value:.0f}', 
                   ha='center', va='center', fontsize=9, weight='bold',
                   rotation=np.degrees(angle)-90 if angle > np.pi/2 and angle < 3*np.pi/2 else np.degrees(angle)+90)
        
        # Formatting
        ax.set_ylim(0, 1.2)
        ax.set_title(f'Polar Area Diagram: {num_col} by {cat_col}', 
                    fontsize=16, pad=30)
        ax.grid(True, alpha=0.3)
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_rlabel_position(0)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show data table
        st.subheader("Data Values")
        polar_df = pd.DataFrame({
            'Category': data.index,
            'Value': data.values,
            'Percentage': (data.values / data.sum() * 100).round(1)
        })
        st.dataframe(polar_df)
        
        # Show statistics
        st.subheader("Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", f"{data.sum():.0f}")
        with col2:
            st.metric("Average", f"{data.mean():.1f}")
        with col3:
            st.metric("Highest", f"{data.max():.0f}")
        with col4:
            st.metric("Lowest", f"{data.min():.0f}")

def show_streamgraphs(df, visualizer):
    st.header("🌊 Streamgraphs")
    
    if len(visualizer.categorical_cols) < 2 or not visualizer.numerical_cols:
        st.warning("Need at least 2 categorical columns and 1 numerical column for streamgraphs.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("Select X-axis (Flow)", visualizer.categorical_cols + visualizer.datetime_cols + ["Index"])
    with col2:
        stack_col = st.selectbox("Select Categories (Streams)", visualizer.categorical_cols)
    with col3:
        value_col = st.selectbox("Select Value Column", visualizer.numerical_cols)
    
    if x_col and stack_col and value_col:
        # Limit categories for readability
        if df[stack_col].nunique() > 8:
            st.warning(f"Too many categories in {stack_col} (>8). Showing top 8.")
            top_cats = df[stack_col].value_counts().head(8).index
            plot_df = df[df[stack_col].isin(top_cats)]
        else:
            plot_df = df
        
        # Prepare data
        if x_col == "Index":
            pivot_df = plot_df.pivot_table(index=plot_df.index, columns=stack_col, 
                                          values=value_col, aggfunc='sum', fill_value=0)
            x_data = pivot_df.index
        else:
            pivot_df = plot_df.pivot_table(index=x_col, columns=stack_col, 
                                          values=value_col, aggfunc='sum', fill_value=0)
            x_data = pivot_df.index
        
        # Create streamgraph
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate baseline for centering (wiggle algorithm approximation)
        data_matrix = pivot_df.values.T
        n_streams = len(data_matrix)
        
        # Simple baseline calculation - center the streams
        baselines = np.zeros_like(data_matrix[0])
        for i in range(n_streams):
            if i == 0:
                baselines = -data_matrix[i] / 2
            else:
                baselines = baselines - data_matrix[i] / 2
        
        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, n_streams))
        
        # Plot streams
        y_bottom = baselines.copy()
        for i, (stream_name, stream_data) in enumerate(pivot_df.items()):
            y_top = y_bottom + stream_data.values
            
            ax.fill_between(x_data, y_bottom, y_top, 
                           color=colors[i], alpha=0.8, label=stream_name)
            
            y_bottom = y_top
        
        # Formatting
        ax.set_xlabel(x_col)
        ax.set_ylabel(value_col)
        ax.set_title(f'Streamgraph: {value_col} by {x_col} and {stack_col}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if x_col in visualizer.datetime_cols or x_col in visualizer.categorical_cols:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show data table
        st.subheader("Stream Data")
        st.dataframe(pivot_df)
        
        # Show stream statistics
        st.subheader("Stream Statistics")
        stream_stats = pivot_df.agg(['sum', 'mean', 'max']).round(2)
        st.dataframe(stream_stats)
        
        # Show flow analysis
        st.subheader("Flow Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Peak Values by Stream:**")
            peak_values = pivot_df.max().sort_values(ascending=False)
            st.dataframe(peak_values.to_frame('Peak Value'))
        with col2:
            st.write("**Total Contribution:**")
            total_contrib = pivot_df.sum().sort_values(ascending=False)
            st.dataframe(total_contrib.to_frame('Total'))

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