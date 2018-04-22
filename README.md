# Detecting Human Emotions Using Natural Language Processing

# [Video Demonstration (YouTube)](https://youtu.be/tMPek5-RY40)

#### The purpose of this project was to develop an application that could be used to detect depression or suicidal ideation in user submitted social media posts. 

## How to use the app.
To run the application, simply enter the following code into your terminal:

```git clone https://github.com/Allenfp/DepressionDetectionNLP```


Once the repository is downloaded enter the following command into your command line:

```python app.py```

Once the application is running, go to the web address shown on the terminal. Thats it!

## Performing analysis
There are two ways to use the app. One is using the website's "Enter comment here..." and pressing submit. The other way is using the app's API. For instructions on using the api [click this link.](https://github.com/Allenfp/DepressionDetectionNLP/blob/master/API_documentation.md)


## Tools used
Data Collection: Python, Praw, Requests, and Pandas.

Analysis: Python, PySpark, VADER Sentiment Analysis, and Pandas.

Visualization: Python, Flask, Wordcloud, WTForms, Bootstrap, HTML, CSS.


## Methodology
The dataset used in this project was constructed by extracting user submitted posts from the r/SuicideWatch and r/CasualConversation subreddits found on Reddit.com.
Data was extracted from Reddit using the Reddit API, and PRAW: The Python Reddit API Wrapper. The Reddit API limits calls to only the 1000 most recent posts on any subreddit. 
To work around this limitation, we performed api calls periodically over a 10 day period for both subreddits. 

Posts were aggregated and checked for duplicates.
In the end our dataset was comprised of 3412 Reddit submissions. These submissions came out to 549,008 words, 32,288 lines, 2,873,664 characters, or about 705 pages of single spaced 12 point Times New Roman text.

The data taken from Reddit was classified according to which subreddit and had VADER Sentiment analysis performed on each submission. A training set of data was created separately from the test set. 
The training set of data was selectively pruned to eliminate exceedingly positive r/SuicideWatch submissions, and exceedingly negative r/CasualConveration submissions. This was done to create higher divergence in the training data. 


After the pruning process was completed, the data was fitted and transformed using a Spark Machine Learning Pipeline that created features based on TF-IDF analysis and the negative sentiment VADER Sentiment analysis and was vectorized.
Using the newly transformed data, a Naive Bayes multinomial classifier model is trained. This model was used to predict whether a given submission from the test set was posted in r/SuicideWatch or r/CasualConversation. Accuracy was determined to be ~90% +/- ~2%.

## Disclaimer 
This website is a project designed to predict emotional sentiments using machine learning. This is in no way a clinical tool, nor should it be used as a replacement for professional or clinical advice. If you are in a state of emergency, please call 911 or/and seek professional support immediately. 



## Flowchart
![alt text](https://github.com/Allenfp/DepressionDetectionNLP/blob/master/wordmap_and_flowchart/Depression%20Detecting%20NLP%20Model.png)

## Website
![alt text](https://github.com/Allenfp/DepressionDetectionNLP/blob/master/DepressionNLPwebsite.png)
