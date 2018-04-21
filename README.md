# Detecting Human Emotions Using Natural Language Processing

#### The purpose of this project was to develop an application that could be used to detect depression or suicidal ideation in user submitted social media posts. 

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

## Flowchart
![alt text](https://github.com/Allenfp/DepressionDetectionNLP/blob/master/wordmap_and_flowchart/Depression%20Detecting%20NLP%20Model.png)

## Website
![alt text](https://github.com/Allenfp/DepressionDetectionNLP/blob/master/DepressionNLPwebsite.png)
