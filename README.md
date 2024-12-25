# Flipkart Customer Sentimental Analysis - Python

## Overview
This repository contains a sentiment analysis project performed on customer reviews scraped from Flipkart's website. The analysis aims to identify patterns and sentiments in customer feedback, leveraging Python for data processing and visualization.

## Project Objective
As a Data Analyst at Flipkart, you have been tasked with gauging customer sentiment towards the iPhone 15 128GB model. The primary goal of this project is to analyze public perception and evaluate customer reactions by performing sentiment analysis on product reviews posted by users. By extracting and processing customer reviews, you will derive insights about the overall sentiment (positive or negative) surrounding the product, which can be useful for decision-making, improving customer experience, and identifying key areas for product improvement.

## Folder Structure
```
Flipkart-Customer-Sentimental-Analysis---Python/
|
|-- datasets/                 # Contains the scraped dataset
|   |-- flipkart_data.csv     # Scraped customer reviews
|
|-- notebooks/                # Jupyter Notebooks for analysis
|   |-- Flipkart-Customer-Sentimental-Analysis.ipynb
|
|-- requirements.txt          # Python dependencies
|
|-- README.md                 # Project documentation (this file)
```

## Installation
To run the project locally, ensure you have Python installed on your machine. Then, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/EthenDcosta5/Flipkart-Customer-Sentimental-Analysis---Python.git
   cd Flipkart-Customer-Sentimental-Analysis---Python
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook to run the analysis:
   ```bash
   jupyter notebook
   ```

## Dataset
The dataset (`flipkart_data.csv`) contains the following columns:
- `review_text`: The review provided by the customer.
- `rating`: The rating given by the customer (1-5).

## Key Python Code

### Data Preprocessing
```python
import pandas as pd

# Load dataset
data = pd.read_csv('datasets/flipkart_data.csv')

# Overview of the dataset
print(data.head())
print(data.info())

# Checking for missing values
print(data.isnull().sum())

# Basic statistics
print(data.describe())
```

### Data Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize rating distribution
sns.countplot(data['rating'], palette='viridis')
plt.title('Rating Distribution')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()
```

## Results
- **EDA Findings**: Distributions of customer ratings and review statistics were visualized.

## Future Work
- Enhance text preprocessing with advanced techniques.
- Explore customer feedback trends over time.

## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For queries or collaborations, reach out to:
- **GitHub**: [EthenDcosta5](https://github.com/EthenDcosta5)

---

Feel free to explore, star the repository, and suggest improvements! 🚀
