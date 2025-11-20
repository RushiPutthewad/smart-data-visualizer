# 🎯 Smart Data Visualizer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Intelligent data visualization platform with 20+ chart types and automated analysis capabilities**

Transform your CSV data into stunning, interactive visualizations with zero configuration. Built for data analysts, researchers, and business professionals who need quick insights from their data.

## ✨ Key Features

### 🚀 **Dual Interface Options**
- **Web Application**: Interactive Streamlit dashboard with real-time controls
- **Command Line Tool**: Automated batch processing for multiple datasets

### 📊 **20+ Professional Chart Types**
| Category | Chart Types |
|----------|-------------|
| **Distribution** | Histograms, Box Plots, Density Plots |
| **Correlation** | Heatmaps, Scatter Plots |
| **Categorical** | Pie Charts, Bar Charts, Count Plots |
| **Advanced** | Treemaps, Sunburst, Waterfall, Funnel |
| **Multi-dimensional** | Bubble Charts, Grouped/Stacked Charts |
| **Specialized** | Polar Area Diagrams, Streamgraphs |
| **Time Series** | Line Charts, Area Charts |

### 🧠 **Intelligent Analysis**
- **Auto Column Detection**: Numerical, categorical, and datetime recognition
- **Smart Chart Selection**: Automatically chooses appropriate visualizations
- **Data Quality Checks**: Missing values, duplicates, and outlier detection
- **Statistical Insights**: Correlation analysis and descriptive statistics
- **Professional Reports**: Automated insights generation

### 🎨 **Interactive Features**
- **Page-based Navigation**: Organized chart categories
- **Dynamic Filtering**: Interactive column selection
- **Multiple Chart Styles**: Grouped, stacked, horizontal/vertical options
- **Real-time Updates**: Instant visualization changes
- **Export Ready**: High-resolution PNG outputs (300 DPI)

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/smart-data-visualizer.git
cd smart-data-visualizer
pip install -r requirements.txt
```

### Web Application (Recommended)
```bash
streamlit run streamlit_app.py
```
Then open `http://localhost:8501` in your browser.

### Command Line Tool
```bash
python intelligent_data_visualizer.py your_data.csv
```

### Python Integration
```python
from intelligent_data_visualizer import IntelligentDataVisualizer

analyzer = IntelligentDataVisualizer('data.csv')
analyzer.run_analysis()
```

## 📁 Project Structure

```
smart-data-visualizer/
├── streamlit_app.py              # 🌐 Web application (main interface)
├── intelligent_data_visualizer.py # 🤖 CLI tool & core library
├── example_usage.py              # 📚 Usage examples
├── requirements.txt              # 📦 Dependencies
├── run_streamlit.bat            # 🪟 Windows launcher
├── README.md                    # 📖 Documentation
└── cleaned_car_sales_dataset_comprehensive.csv # 📊 Sample data
```

## 🎯 Use Cases

### 📈 **Business Analytics**
- Sales performance analysis
- Customer segmentation
- Market trend visualization
- KPI dashboards

### 🔬 **Research & Academia**
- Exploratory data analysis
- Statistical reporting
- Publication-ready charts
- Dataset profiling

### 💼 **Data Science**
- Feature correlation analysis
- Data quality assessment
- Model input visualization
- Results presentation

## 🛠️ Technical Specifications

### **Core Technologies**
- **Frontend**: Streamlit (Interactive web interface)
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Supported Formats**: CSV files

### **System Requirements**
- Python 3.8+
- 4GB RAM (recommended)
- Modern web browser (for Streamlit interface)

### **Dependencies**
```
streamlit >= 1.28.0
pandas >= 1.5.0
numpy >= 1.21.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
```

## 📊 Chart Gallery

### **Distribution Analysis**
- **Histograms**: Frequency distributions with customizable bins
- **Box Plots**: Quartile analysis with outlier detection
- **Density Plots**: Smooth probability distributions

### **Advanced Visualizations**
- **Treemaps**: Hierarchical data representation
- **Sunburst Charts**: Multi-level categorical relationships
- **Waterfall Charts**: Step-by-step value changes
- **Funnel Charts**: Conversion rate analysis

### **Multi-dimensional Analysis**
- **Bubble Charts**: 4-variable relationships (X, Y, size, color)
- **Grouped/Stacked Charts**: Category comparisons
- **Polar Area Diagrams**: Circular data representation

## 🎨 Output Examples

### **Automated Reports**
- `analysis_insights.txt` - Comprehensive findings report
- `summary_statistics.csv` - Statistical summary table
- Multiple PNG charts (300 DPI) - Publication-ready visualizations

### **Interactive Features**
- Real-time chart updates
- Dynamic column selection
- Multiple aggregation options (Count, Sum, Average, Min/Max, Percentages)
- Responsive design for all screen sizes

## 🚀 Advanced Usage

### **Batch Processing**
```bash
# Process multiple files
for file in *.csv; do
    python intelligent_data_visualizer.py "$file"
done
```

### **Custom Analysis**
```python
# Advanced customization
visualizer = IntelligentDataVisualizer('data.csv')
visualizer.load_and_analyze_data()
visualizer.detect_column_types()

# Generate specific chart types
visualizer.create_correlation_analysis()
visualizer.create_categorical_analysis()
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
git clone https://github.com/yourusername/smart-data-visualizer.git
cd smart-data-visualizer
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Visualization powered by [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- Data processing with [Pandas](https://pandas.pydata.org/)

## 📞 Support

- 📧 **Email**: support@smartdatavisualizer.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/smart-data-visualizer/issues)
- 📚 **Documentation**: [Wiki](https://github.com/yourusername/smart-data-visualizer/wiki)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the data community

</div>