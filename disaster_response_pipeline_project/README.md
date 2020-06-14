## Disaster Response Pipeline Project

### Project Motivation

This project is one of the portfolio projects from Udacity's Data Science Nanodegree Program.

Main goal of the project was to create a classification model which automatically labels disaster response messages with 36 different category labels.

A simple web app is provided to test how the model works.

### Installation and Use

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*

The following steps are needed to run the app:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
      
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves
      
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

You can enter a message in the box, press the "Classify Message" button and the classifier will color the labels it finds most suitable.

### File Structure

Here's how the project structure looks like after installation:

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

### Licencing, Authors

Web app code, csv files and code structure of process_data.py and train_classifier.py were provided by Udacity.com. The rest was done by me using the skills I've developed throughout the program.

### Github Repo

Here's a [link](https://github.com/Nikalozj/Disaster-Response.git) to my github repo.