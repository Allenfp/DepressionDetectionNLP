# Import dependencies
from flask import Flask, render_template, jsonify, request, redirect, flash
from sqlalchemy import func
# from model import session
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

#################################################
# Flask Setup
#################################################
DEBUT = True
app = Flask(__name__)
app.config.from_object(__name__)

#################################################
# Routes
#################################################

class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])

# Main route
@app.route('/')
def home():
    form = ReusableForm(request.form)

    print(form.errors)
    if request.method == 'POST':
        name = request.form['inputText']
        print(name)

        if form.validate():
            flash(name)

        else:
            flash('All the form fields are required.')
            
    return render_template('index.html', form=form)

# List of other points

if __name__ == "__main__":
    app.run(debug=True)
