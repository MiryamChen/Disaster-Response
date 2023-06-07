import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import pickle

def load_data(database_filepath):
    """
    loads the cleaned dataframe from the database.
    Args:
    database_filepath(str): file path of the sqlite database
    return: 
    X(pandas dataframe): messages(features)
    y(pandas dataframe): targets
    categories(list): list of categorical columns
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('dis_response_df', engine)
    X = df['message']
    y = df.iloc[:, 3:]
    category_names = y.columns.tolist() 
    return X, y, category_names

def tokenize(text):
    """
    Tokenizes the raw text data.
    Args:
    """
    text = text.lower() #lowercase everything
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) #remove punctuations
    words = word_tokenize(text) #tokenize the text
    words = [w for w in words if w not in stopwords.words("english")] #remove stop words
  
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in words:
             clean_token = lemmatizer.lemmatize(token).strip()
             clean_tokens.append(clean_token)
    return clean_tokens


def build_model(X, y):
    """
    this machine learning piepeline takes in message and targets (X, y) to train
    """
    #define pipeline, renamed pipeline as model, because I commented out the Gridserach for speed
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('multi_dt', MultiOutputClassifier(DecisionTreeClassifier()))
])
    #define parameters to run gridsearch, commenting out Gridsearch for speed
    """parameters ={
    'vect__max_features': [300, 500],  # Number of features in CountVectorizer
    'tfidf__use_idf': [True, False],  # Whether to use IDF in TfidfTransformer
    'multi_dt__estimator__max_depth': [None, 3, 6]}"""

    #define Gridsearch model
    #commenting out Gridsearch: model = GridSearchCV(pipeline, param_grid = parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        y_pred_cols = y_pred[:, i]
        y_test_cols = Y_test.iloc[:, i]
    
    report = classification_report(y_test_cols, y_pred_cols)
    
    print(report)


def save_model(model, model_filepath):
    """
    save a machine learning model to a file
    Args:
    model(object): the trained machine learning model object.
    model_filepath(str): the file path(incl file name) where the model will be saved
    Retuns:
    None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
        


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X, y)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()