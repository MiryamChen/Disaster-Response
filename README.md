# Disaster-Response

This project builds a website that classifies incoming texts into different disaster response categories. The objective is to facilitate the processing of incoming messages during disasters. 

The website is powered by a model that does the text classification behind. It also includes 3 graphs to visually check messages per genre, top 10 categories with most messages, and the bottom 10 categories with least messages received. 

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


# Files included: 
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

- work books
|- ETL Pipeline Preparation.ipynb #Jupyter notebook file for testing codes
|- ML Pipeline Preparation.ipynb #Jupyter notebook file for testing codes

# List of libraries used
Pandas
sqlalchemy
sys
nltk
re
pickle

# Aknowledgement
Appreciations to the Appen organization for providing the data for this project. 

