# Social Media Emotional Analysis API

## Retrieve Sentiments

# URL - 

/api/v1.0/<your_string_here>

# URL PARAMS - 

None. Simply enter your string at the end of the API endpoint for a jsonified 
analysis.

# SAMPLE CALL -

127.0.0.1:5000/api/v1.0/I am having a lovely evening despite the weather.

# SAMPLE RESULT - 

    {
    "prediction": 1.0, 
    "probability_depression_0": 6.908625777155489e-19, 
    "probability_normal_1": 1.0, 
    "vader_senti_neg": 0.388, 
    "vader_senti_pos": 0.0
    }

# DESCRIPTION OF RESULT PARAMS

*"prediction" is a binary return value. If 0, depression or suicidal ideation is detected.
*"probability_depression_0" is a probability (between 0 and 1) of the model's confidence that
    the submitted text contains evidence of depression or suicidal ideation.
*"probability_normal_1" is a probability (between 0 and 1) of the model's confidence that 
    the submitted text does not contain evidence of depression or suicidal ideation.
*"vader_senti_neg" Ratio of proportions of text that fall in the negative category.
*"vader_senti_pos" Ratio of proportions of text that fall in the negative category.
