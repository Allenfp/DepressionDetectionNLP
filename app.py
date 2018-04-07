# Import dependencies
from flask import Flask, render_template, jsonify, request, redirect
from sqlalchemy import func
from model import session


#################################################
# Flask Setup
#################################################

app = Flask(__name__)

#################################################
# Routes
#################################################


# Main route
@app.route('/')
def home():
    return render_template('index.html')

# List of other points

if __name__ == "__main__":
    app.run(debug=True)
