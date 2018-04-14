from flask import Flask, render_template, flash, request, jsonify
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes

#Start Spark Session
conf = SparkConf().setAppName('original_NB').setMaster("local")
conf.set("spark.executor.memory", "4g") #This will take at least 4G to run.
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

#############################
####### TRAINING DATA #######
#############################

#Build Pandas DataFrame
train_df = pd.read_csv("final_training_data.txt", delimiter="\t")
train_df["pos"] = ""
train_df["neg"] = ""
train_df["label"] = ""
train_df.head()

#Perform Sentiment Analysis on Training Data and Populate Pandas DataFrame
analyzer = SentimentIntensityAnalyzer()

for index, row in train_df.iterrows():

    scores = analyzer.polarity_scores(row[0])
    pos = scores['pos']
    neg = scores['neg']
    row[2] = pos 
    row[3] = neg
    if row.subreddit == "sw":
        row.label = 0
    elif row.subreddit == "cc":
        row.label = 1
#train_df

#Eliminates negative cc posts and positive sw posts. Creates better defined training set.
for index, row in train_df.iterrows():
    if row.neg > 0.1 and row.subreddit == "cc":
        train_df.drop(index, inplace=True)
    elif row.pos > 0.1 and row.subreddit == "sw":
        train_df.drop(index, inplace=True)    
    else:
        pass
    
#Put Pandas DataFrame into Spark DataFrame
tr_spark_df = spark.createDataFrame(train_df)

#################################
####### TRAIN & FIT DATA ########
#################################

# Define the PipeLine Variables
tokenizer = Tokenizer(inputCol="combined", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
hashingTF = HashingTF(inputCol="stop_tokens", outputCol='hash_token')
idf = IDF(inputCol='hash_token', outputCol='idf_token')
clean_up = VectorAssembler(inputCols=['idf_token','neg'], outputCol='features')

# Define the Pipeline Variable
data_prep_pipeline = Pipeline(stages=[tokenizer, stopremove, hashingTF, idf, clean_up])

# Fit and Transform the Training and Testing Spark Dataframes.

#First the Cleaning
cleaner_train = data_prep_pipeline.fit(tr_spark_df)
cleaned_train = cleaner_train.transform(tr_spark_df)

#Note, keeping this here as a tool in case we need to tune model later.
(training, x) = cleaned_train.randomSplit([1.0, 0.0]) #Uses Entire Training Set (After vadersentiment pruning)

# Create a Naive Bayes model and fit training data
nb = NaiveBayes(smoothing=1.0, modelType='multinomial')
sub_predictor = nb.fit(training)

def check_string(test_string):

    #Create Single Input Pandas DataFrame.
    d = {'combined': test_string, 'pos':"", 'neg':""}
    df = pd.DataFrame(d, index=[0])

    #Analyze Sentiment and Populate DataFrame. 
    analyzer = SentimentIntensityAnalyzer()
    for index, row in df.iterrows():
        scores = analyzer.polarity_scores(row[0])
        pos = scores['pos']
        neg = scores['neg']
        row[1] = pos 
        row[2] = neg

    #Convert to Spark DataFrame.
    sdf = spark.createDataFrame(df)

    #Fit & Clean Spark DataFrame 
    cleaned_sc = cleaner_train.transform(sdf)
    cleaned_sc.show()
    print(type(sub_predictor))

    #Perform Prediction and Print results.
    pc_results = sub_predictor.transform(cleaned_sc)
    pc_results.select('probability', 'prediction').show(truncate=False)
    print("1 = Casual Conversation, 0 = Depression/Suicial Ideation")

    response = ""
    prediction = pc_results.select('prediction').rdd.map(lambda row : row[0]).collect()[0]
    probability = pc_results.select('probability').rdd.map(lambda row : row[0]).collect()[0]
    probability_sad = probability[0]
    probability_normal = probability[1]

    if prediction == 0 and probability_sad >= .75:
        response = f"YIKES! There is ample evidence to indicate sadness and/or depression.  Reliability: {round(probability_sad*100)}% "
    elif prediction == 1 and probability_normal >= .75:
        response = f"Everything is looking good over here! Reliability: {round(probability_normal*100)}% "
    else:
        response = f"Sorry, the analysis is not confident enough to render a reliabile decision with the given text. Please try a longer submission."
            

    return response



def check_string_api(test_string):

    #Create Single Input Pandas DataFrame.
    d = {'combined': test_string, 'pos':"", 'neg':""}
    df = pd.DataFrame(d, index=[0])

    #Analyze Sentiment and Populate DataFrame. 
    analyzer = SentimentIntensityAnalyzer()
    for index, row in df.iterrows():
        scores = analyzer.polarity_scores(row[0])
        pos = scores['pos']
        neg = scores['neg']
        row[1] = pos 
        row[2] = neg

    #Convert to Spark DataFrame.
    sdf = spark.createDataFrame(df)

    #Fit & Clean Spark DataFrame 
    cleaned_sc = cleaner_train.transform(sdf)
    cleaned_sc.show()
    print(type(sub_predictor))

    #Perform Prediction and Print results.
    pc_results = sub_predictor.transform(cleaned_sc)
    pc_results.select('probability', 'prediction').show(truncate=False)
    print("1 = Casual Conversation, 0 = Depression/Suicial Ideation")

    
    prediction = pc_results.select('prediction').rdd.map(lambda row : row[0]).collect()[0]
    probability = pc_results.select('probability').rdd.map(lambda row : row[0]).collect()[0]
    vader_senti_pos = sdf.select('pos').rdd.map(lambda row : row[0]).collect()[0]
    vader_senti_neg = sdf.select('neg').rdd.map(lambda row : row[0]).collect()[0]
    probability_sad = probability[0]
    probability_normal = probability[1]

    response = {
        'prediction' : prediction,
        'probability_depression_0' : probability_sad,
        'probability_normal_1' : probability_normal,
        'vader_senti_pos' : vader_senti_pos,
        'vader_senti_neg' : vader_senti_neg
    }


    return response



# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
 
class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])
 
 
@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
 
    
    if request.method == 'POST':
        name=request.form['name']
        print(name)
 
        if form.validate():
            check_string(name)
            flash(check_string(name))
        else:
            flash('Error: All the form fields are required. ')
 
    return render_template('index.html', form=form)

@app.route("/api/v1.0/<_string>", methods=['GET'])
def api(_string):

    return jsonify(check_string_api(_string))

 
if __name__ == "__main__":
    app.run()
