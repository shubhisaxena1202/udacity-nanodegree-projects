from sre_parse import Tokenizer
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

def load_data(database_filepath):
    '''
    Load data from database

    Parameters: database_filepath - File path to database

    Returns:
    X: training features
    Y: target
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y
    # pass


def tokenize(text):
    '''
    Tokenizes and lemmatizes text

    Parameters: text - text to be cleaned

    Returns:
    clean_tokens : list of cleaned tokens
    '''
    token = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens=[]
    for x in token:
        clean = lemmatizer.lemmatize(x).lower().strip()
        clean_tokens.append(clean)
    return clean_tokens
    # pass




def build_model():
    '''
    Create Classifier and tune model parameters using GridSearchCV

    Returns:
    CV: gridSearch Classifier
    '''
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    params = {
#         'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    cv= GridSearchCV(pipeline,param_grid=params,verbose=3)
    return cv
    # pass



def evaluate_model(model, X_test, Y_test):
    '''
    Evaluate Model and make predictions

    Paramters : Model - RandomForest Classifier
    X_test - features test
    Y_test - target test

    Returns:
    Prints Classification report
    '''
    y_pred = model.predict(X_test)
    for i,col in enumerate(Y_test):
        print(col,classification_report(Y_test[col],y_pred[:,i]))
    # pass


def save_model(model, model_filepath):
    # pass
    pickle.dump(model,open(model_filepath,'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
