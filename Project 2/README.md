# Disaster Response Pipeline Project


### Objective:
In this application, we analyze disaster data to build a model for an API that classifies disaster messages. The project includes a web app where an emergency worker can input a new message and get classification results. The web application can also display visualizations of the data.

### Components

There are three components in this project.

1. ETL Pipeline

A Python script, process_data.py, which is a data cleaning pipeline.

2. ML Pipeline

A Python script, train_classifier.py, which is a machine learning pipeline.

3. Flask Web App

The project includes a web app where an emergency worker can input a new message to get classification results.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
