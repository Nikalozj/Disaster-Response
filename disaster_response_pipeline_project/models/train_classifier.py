import sys
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):

    """
	Loads data to a dataframe and splits it into variables X and Y 

	Parameters:
		database_filepath -- path of the database

	Returns:
		X - list of messages to be processed
		Y - categories
		Y.columns.values - list of column names

	"""

    #Read data to df
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM {}'.format(database_filepath.split("/",1)[1][:-3]), engine)
    
    #Create X and Y
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    
    return X, Y, Y.columns.values


def tokenize(text):

	""" Takes text and returns cleaned list of tokens """

	#Create tokens' list
    tokens = word_tokenize(text)

    #Create lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #Clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():

    """ Creates a pipeline, tunes it with GridsearcgCV and returns a tuned model"""


    #create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier())
    ])
    
    #Create parameters for GridsearchCV
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'moc__estimator': (RandomForestClassifier(), ExtraTreesClassifier())
    }
    

    #Optimize the model
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """
	Predicts Y values using the model and prints classification report for each column

	Parameters:
		model -- model to be tested
		X_test -- X test values
		Y_test -- Y test values
		category_names -- Y column names

    """

    #predict values using the model
    Y_pred = model.predict(X_test)
    
    #iterate through each column of Y test and print classification_report for each of them
    for i in range(0, len(Y_test.columns.values)):
        print("-----------------")
        print(category_names[i])
        print("-----------------")
        print(classification_report(Y_test.values.transpose()[i], Y_pred.transpose()[i]))


def save_model(model, model_filepath):

    """ Save model as pickle file """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:

    	#Put database and model filepaths into variables
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        #Load data
        X, Y, category_names = load_data(database_filepath)

        #Apply train_test_split to the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        #Create model
        print('Building model...')
        model = build_model()
        
        #Fit the model
        print('Training model...')
        model.fit(X_train, Y_train)
        
        #Evaluate the model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        #Save the model
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