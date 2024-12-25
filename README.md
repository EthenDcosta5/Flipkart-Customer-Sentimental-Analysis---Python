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

```python
# Plotting ratings vs average polarity
plt.figure(figsize=(10, 6))
sns.boxplot(x='Average_Polarity', y='Ratings', data = new_data, hue = 'Average_Polarity', palette='coolwarm')
plt.title('Ratings vs Average Polarity')
plt.xlabel('Average Polarity')
plt.ylabel('Ratings')
plt.xticks(rotation=90)
plt.savefig('charts/ratings-polarity.jpg')
plt.show()
```
![Ratings vs Average Polarity](https://github.com/EthenDcosta5/Flipkart-Customer-Sentimental-Analysis---Python/blob/main/charts/ratings-polarity.jpg)

**<u>Correlation</u>:**
- **Higher sentiment polarities align closely with higher ratings** (e.g., 4.5–5), as evident from the clustering and color gradient.

**<u>Neutral Reviews</u>:**
- **Neutral categories show a balanced spread across various ratings**, indicating less agreement between sentiment and star ratings.

**<u>Negative Reviews</u>:**
- **Negative and extremely negative reviews often have lower average ratings** but may still exhibit variability due to subjective interpretation by reviewers.

```python
from wordcloud import WordCloud

positive_reviews = []
negative_reviews = []

# Classify the positive & negative reviews separately
for i in range(len(new_data)):
    if new_data.iloc[i]['Sentiment_Class'] == 'positive' or new_data.iloc[i]['Sentiment_Class'] == 'extremely positive':
        positive_reviews.append(new_data.iloc[i]['Review'])
    elif new_data.iloc[i]['Sentiment_Class'] == 'negative' or new_data.iloc[i]['Sentiment_Class'] == 'extremely negative':
        negative_reviews.append(new_data.iloc[i]['Review'])

# Assign random positive and negative reviews to create a cloud map
pos = positive_reviews[240]
neg = negative_reviews[1]

# Generate word clouds for positive and negative reviews
positive_wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Greens").generate(pos)
negative_wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(neg)

# Plot the word clouds
plt.figure(figsize=(16, 8))

# Positive reviews word cloud
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews Word Cloud", fontsize=16)

# Negative reviews word cloud
plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews Word Cloud", fontsize=16)
plt.tight_layout()
plt.savefig('charts/word-cloud.jpg')
plt.show()
```
![Word Cloud](https://github.com/EthenDcosta5/Flipkart-Customer-Sentimental-Analysis---Python/blob/main/charts/word-cloud.jpg)

<u> **Word Cloud Description:** </u>

The above image displays two word clouds generated from customer reviews:

1. **Positive Reviews Word Cloud** (left side, green color):  
   Highlights frequently mentioned positive words like **"mobile," "first," "performance,"** and **"camera,"** indicating attributes appreciated by customers.

2. **Negative Reviews Word Cloud** (right side, red color):  
   Features prominent negative terms such as **"fast," "giving," "refresh,"** and **"shame,"** representing commonly cited issues or complaints.

```python
# Calculate the length of the sentences by calculating the number of words in the review sentence
new_data['Review_Length'] = new_data['Review'].apply(lambda x: len(x.split()))

# Box Plot for Review Length by Sentiment
plt.figure(figsize=(8, 6))
sns.boxplot(x='Sentiment_Class', y='Review_Length', data=new_data, hue = 'Sentiment_Class', palette='Set2')
plt.title('Review Length vs Sentiment', fontsize=14)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Review Length (Number of Words)', fontsize=12)
plt.savefig('charts/length-sentiment-corr.jpg')
plt.show()
```
![Review Length vs Sentiment](https://github.com/EthenDcosta5/Flipkart-Customer-Sentimental-Analysis---Python/blob/main/charts/length-sentiment-corr.jpg)

**Observations:**

<u> **Positive Sentiment:** </u>
- **Has the largest variability in review length**, with several outliers.
- **The median is higher** compared to other categories.

<u> **Extremely Positive Sentiment:** </u>
- **Has the shortest review lengths overall**, with a compact distribution and fewer outliers.

<u> **Neutral Sentiment:** </u>
- **Shows a small range of review lengths**, similar to the "Extremely Positive" category.

<u> **Negative Sentiment:** </u>
- **Exhibits a moderate range of review lengths.**
- **The median review length is smaller** than "Positive" but larger than "Extremely Positive" and "Neutral."

<u> **Interpretation:** </u>
- **Positive reviews tend to be more detailed (longer)** compared to other sentiments.
- **Extremely positive and neutral reviews are often brief**.
- **Negative reviews have varying lengths** but are generally less wordy than positive reviews.

## Sentimental Analysis Report
#### <b>Sentiment Analysis Report: Flipkart Customer Reviews for iPhone 15 128GB</b>

##### 1. Overview of the Data Collection and Cleaning Process
- **Data Source**: Customer reviews were collected from Flipkart for the iPhone 15 128GB model through web scraping with the help of libraries like Selenium and BeautifulSoup .
- **Preprocessing**:
  - Reviews were cleaned by removing irrelevant characters, converting cases, and unnecessary spaces.
  - Text was tokenized to standardize the input for analysis.
  - Sentiments were classified into categories (e.g., positive, extremely positive, neutral, negative, extremely negative) using sentiment analysis techniques.

##### 2. Sentiment Analysis Results
- **Sentiment Distribution**:
  - A majority of reviews were positive, followed by extremely positive ones, as evident from the sentiment distribution graph.
  - Neutral and negative sentiments accounted for a significantly smaller proportion of the reviews.
- **Average Sentiment Per Rating**:
  - Higher star ratings were consistently associated with positive and extremely positive sentiment.
  - Lower star ratings correlated with neutral or negative sentiments, pinpointing dissatisfaction in these reviews.

##### 3. Insights
##### Positive Highlights
- Customers appreciated the **design, camera quality, and overall performance** of the iPhone 15.
- **Battery life improvements** were a common positive theme.
  
##### Common Issues
- Neutral and negative sentiments highlighted **pricing concerns** and occasional issues with **delivery or packaging**.
- A few reviews mentioned **compatibility issues** with accessories or software glitches.

##### 4. Recommendations
##### Product Improvements
- Consider addressing minor software glitches highlighted by users.
- Investigate compatibility issues with certain accessories to ensure a seamless customer experience.

##### Marketing Focus
- Highlight positive aspects like **camera performance**, **battery life**, and the **sleek design** in promotional campaigns.
- Address pricing concerns through **EMI options**, **exchange offers**, or limited-time discounts to make the product more accessible.

##### Operational Enhancements
- Improve **delivery processes** to minimize complaints about packaging or delays.
- Monitor **customer feedback** closely to resolve emerging issues quickly.

## Libraries and Tools used

#### Selenium: 
For automating the web scraping process.

#### BeautifulSoup: 
For parsing HTML and extracting review details.

#### Pandas: 
For data cleaning, processing, and analysis.

#### TextBlob: 
For performing sentiment analysis on the review text.

#### Matplotlib/Seaborn: 
For visualizations like sentiment distribution and word clouds.

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
