# AI Data Analyst

A Python-based AI-powered data analysis tool designed to automate data exploration, visualization, and insights generation.

## 📋 Table of Contents

- [Overview](#overview)
- [Live Website](#live-website)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

AI Data Analyst is an intelligent tool that leverages artificial intelligence and machine learning to analyze datasets, generate insights, and create visualizations automatically. It simplifies the data analysis process, making it accessible to both beginners and experienced data scientists.

## 🌐 Live Website

🔗 Visit my portfolio here: https://aidataanalyst-pibek8udhacksdpakgi2nu.streamlit.app/

## ✨ Features

- **Automated Data Exploration**: Quickly understand your dataset structure and statistics
- **Intelligent Data Cleaning**: Detect and handle missing values, outliers, and data inconsistencies
- **Statistical Analysis**: Generate comprehensive statistical summaries
- **Data Visualization**: Create meaningful charts and graphs automatically
- **Insight Generation**: AI-powered insights and trend detection
- **Pattern Recognition**: Identify correlations and patterns in your data
- **Interactive Reports**: Generate detailed analysis reports

## 📦 Requirements

- Python 3.8 or higher
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- jupyter (optional, for notebook support)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/anandanbu/AI_Data_Analyst.git
cd AI_Data_Analyst
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💡 Usage

### Basic Example

```python
from ai_data_analyst import DataAnalyst

# Load your dataset
analyst = DataAnalyst('path/to/your/data.csv')

# Generate analysis
analyst.explore()
analyst.analyze()
analyst.visualize()
analyst.generate_report('report.html')
```

### Command Line Usage

```bash
python analyze.py --input data.csv --output report.html
```

## 📁 Project Structure

```
AI_Data_Analyst/
├── README.md
├── requirements.txt
├── setup.py
├── ai_data_analyst/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── visualizer.py
│   ├── preprocessor.py
│   └── utils.py
├── examples/
│   └── sample_analysis.py
└── tests/
    └── test_analyzer.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or support, please open an issue on the [GitHub repository](https://github.com/anandanbu/AI_Data_Analyst).

---

**Happy analyzing! 🎉**
