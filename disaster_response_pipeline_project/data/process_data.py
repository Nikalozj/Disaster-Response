import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

	#create dataframes from csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge dataframes
    df = messages.merge(categories, on='id', how='left')
    
    return df

def clean_data(df):
    
    #create df of 36 individual columns from 'categories' column, which is delimited by ';'
    categories = df['categories'].str.split(';', expand=True)
    
    #create column names from the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[:len(x)-2])
    categories.columns = category_colnames
    
    #convert category values to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[len(x)-1 : ])

        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x : int(x))
        
    #drop 'categories' column from df
    df.drop(columns = "categories", inplace = True)
    
    #concatenate 'categories' df to original df
    df = pd.concat([df, categories], axis=1)
    
    #drop duplicates
    df.drop_duplicates('original', inplace=True)
    
    return df


def save_data(df, database_filename):
    
    #save data as sql
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename.split("/",1)[1][:-3], engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()