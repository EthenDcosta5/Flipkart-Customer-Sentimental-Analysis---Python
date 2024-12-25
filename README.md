# Flipkart Customer Sentimental Analysis - Python

## Overview
This repository contains a sentiment analysis project performed on customer reviews scraped from Flipkart's website. The analysis aims to identify patterns and sentiments in customer feedback, leveraging Python for data processing and visualization.

## Project Objective
As a Data Analyst at Flipkart, you have been tasked with gauging customer sentiment towards the iPhone 15 128GB model. The primary goal of this project is to analyze public perception and evaluate customer reactions by performing sentiment analysis on product reviews posted by users. By extracting and processing customer reviews, you will derive insights about the overall sentiment (positive or negative) surrounding the product, which can be useful for decision-making, improving customer experience, and identifying key areas for product improvement.

## Dataset
The dataset (`flipkart_data.csv`) contains the following necessary columns:
- `review_text`: The review provided by the customer.
- `rating`: The rating given by the customer (1-5).

## Key Python Code

### Data Collection (Web Scraping)
```python
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Create empty lists to store the user data such as Name, City, Date of Purchase, Review & Rating
Names = []
Cities = []
Dates = []
Reviews = []
Ratings = []

# Assign the url of the flipkart website and use selenium to scrape data
url = """https://www.flipkart.com/apple-iphone-15-blue-128-gb/product-reviews/itmbf14ef54f645d?pid=MOBGTAGPAQNVFZZY&lid=LSTMOBGTAGPAQNVFZZYQRLPCQ&marketplace=FLIPKART"""
driver = webdriver.Chrome()
driver.get(url)


while len(Names) < 320:

    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Scrape names
    temp_names = soup.find_all("p", {"class": "_2NsDsF AwS1CA"})
    for name in temp_names:
        Names.append(name.text)

    # Scrape cities
    temp_cities = soup.find_all("p", {"class": "MztJPv"}) 
    for city in temp_cities:
        Cities.append(city.text)

    # Scrape dates
    temp_dates = soup.find_all("p", {"class": "_2NsDsF"}) 
    for date in temp_dates:
        Dates.append(date.text)
    Actual_Dates = Dates[1::2]

    # Scrape reviews
    temp_reviews = soup.find_all("div", {"class": "ZmyHeo"})
    for review in temp_reviews:
        Reviews.append(review.text)

    # Scrape ratings
    temp_ratings = soup.find_all("div", class_ = "XQDdHH Ga3i8K")
    for ratings in temp_ratings:
        Ratings.append(ratings.text)

    # Try to click the "Next" button
    try:
        next_button = driver.find_element(By.XPATH, "//span[text()='Next']")
        next_button.click()
        time.sleep(5)
    except:
        break

# Combine data into a DataFrame
data = pd.DataFrame({
    "Name": Names[:-1],
    "City": Cities[:-1],
    "Date": Actual_Dates[:-1],
    "Review": Reviews[:-1],
    "Ratings": Ratings
})

# Save to a CSV files
data.to_csv("flipkart_reviews_2.csv", index=False)
```


### Data Cleaning and Preprocessing
```python
# Assign the scraped dataset(csv file) to a dataframe
new_data = pd.read_csv('flipkart_reviews.csv')
new_data

# Check the basic info of the dataframe
new_data.info()

# Check valye counts of the Name column
new_data['Name'].value_counts()

# Drop the duplicates from the dataframe
new_data = new_data.copy()
new_data = new_data.drop_duplicates()
new_data

# Convert the Name column data into Title Case
new_data['Name'] = new_data['Name'].str.title()
new_data.head()

# Clean data of City column by removing unwanted characters/ part of string
new_data['City'] = new_data['City'].str.replace("Certified Buyer, ", "", regex=False).str.strip()
new_data.head()

# Clean data of Review column by removing unwanted characters/ part of string and converting to lowercase
new_data['Review'] = new_data['Review'].str.lower().str.replace("read more", "", regex=False)
new_data.head()
```

### Sentiment Analysis
```python
# Import libraries for Sentimental analysis of review sentences 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Create a column called Reviews_t that stores tokenized sentences from the Review column using the sent_tokenize function.
new_data["Reviews_t"] = new_data['Review'].apply(sent_tokenize)
new_data

# Import mean from statistics for basic statistics
from statistics import mean

# Function created for assigning Polarity to the Reviews_t column
def get_polarity(sentences):
    return [TextBlob(sentence).sentiment.polarity for sentence in sentences]

# Calls get_polarity function on the Reviews_t column to assign polarity
new_data['Polarity'] = new_data['Reviews_t'].apply(get_polarity)

# Function created to calculate the average polarity of each review (Average of polarity for each sentences in a review)
def calculate_average_polarity(polarities):
    return mean(polarities) if polarities else 0

# Calls calculate_average_polarity function on the Polarity column to assign the average polarity for each review
new_data['Average_Polarity'] = new_data['Polarity'].apply(calculate_average_polarity)
new_data['Average_Polarity'] = new_data['Average_Polarity'].round(2)
new_data.head(10)

# Function to assign the Class to the Polarity
def sentiment_class(polarity):
    if polarity > 0.75:
        return 'extremely positive'
    elif 0 < polarity <= 0.75:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    elif -0.75 <= polarity < 0:
        return 'negative'
    else:
        return 'extremely negative'

# Calls sentiment_class function on the Average_Polarit column to assign the sentiment class
new_data['Sentiment_Class'] = new_data['Average_Polarity'].apply(sentiment_class)

new_data.head()

# Calculates and prints the overall average polarity score of the entire dataset of reviews
polarity_score = new_data['Average_Polarity'].mean().round(2)
print(f'Average Polarity Score : {polarity_score}')
if polarity_score > 0.75:
        print('The Average Polarity Score is Extremely Positive')
elif 0 < polarity_score <= 0.75:
    print('The Average Polarity Score is Positive')
elif polarity_score == 0:
    print('The Average Polarity Score is Neutral')
elif -0.75 <= polarity_score < 0:
    print('The Average Polarity Score is Negative')
else:
    print('The Average Polarity Score is Extremely Negative')
```

### Data Visualization and Insights
```python
# Imports libraries for visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Plots figure for Sentiment Distribution based on Sentiment Category
plt.figure(figsize=(10, 6))
sns.histplot(x=new_data.Sentiment_Class, color='green')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.savefig('charts/sentiment-distribution.jpg')
plt.show()
```
![Sentiment-Distribution](https://github.com/EthenDcosta5/Flipkart-Customer-Sentimental-Analysis---Python/blob/main/charts/sentiment-distribution.jpg)
<u>**Sentiment Distribution**</u>

This bar chart displays the distribution of sentiment categories within a dataset. The x-axis represents different sentiment categories, while the y-axis represents the frequency of occurrences in each category. The categories include:

1. **Positive**: This category has the highest frequency, with over 200 instances.
2. **Extremely Positive**: This category comes next, with a significantly lower frequency compared to "Positive".
3. **Neutral**: This category has a much smaller frequency than the previous two.
4. **Negative**: This category has the lowest frequency.

The chart indicates a clear bias towards positive sentiments in the dataset, with "Positive" being the dominant category, followed by "Extremely Positive". Neutral and negative sentiments are comparatively rare.


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
- **LinkedIn**: [EthenDcosta](https://linkedin.com/in/ethendcosta)
- **Mail**: [Ethendcosta5@gmail.com](mailto:ethendcosta5@gmail.com)

---

Feel free to explore, star the repository, and suggest improvements! 🚀
