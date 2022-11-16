'''Aquire and prep Zillow data'''
import pandas as pd
import numpy as np
import os
import env
from env import host, user, pwd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

def get_db_url(database):
    return f'mysql+pymysql://{user}:{pwd}@{host}/{database}'
    '''
    Function reads in credentials from env.py file of the user and returns zillow data.
    '''

def get_properties_2017():
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        query = '''
         SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
            taxvaluedollarcnt, yearbuilt, regionidcounty, fips 
        FROM properties_2017
        JOIN predictions_2017
        ON properties_2017.parcelid = predictions_2017.parcelid
        JOIN propertylandusetype
        ON properties_2017.propertylandusetypeid = propertylandusetype.propertylandusetypeid
        WHERE transactiondate LIKE "2017" AND propertylandusedesc IN ('Single Family Residential' , 'Inferred Single Family Residential')
        '''
        df = pd.read_sql(query,get_db_url('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

def wrangle_zillow():
    '''
    Read zillow into a pandas DataFrame from mySQL
    drop columns, drop any rows with Null values, 
    convert columns data types accordingly,
    return cleaned zillow DataFrame.
    '''
    # Acquire data 
    df = get_properties_2017()

    # Drop all rows with NaN values.
    df = df.dropna()
    
    # Convert to correct datatype
    df['yearbuilt'] = df.yearbuilt.astype(int)
    
    # rename columns
    
    df = df.rename(columns={'yearbuilt':'year_built'})
    df = df.rename(columns={'bedroomcnt':'bed_rooms'})
    df = df.rename(columns={'bathroomcnt':'bath_rooms'})
    df = df.rename(columns={'calculatedfinishedsquarefeet':'finished_sqft'})
    

    df = df[df.finished_sqft < 15_000]
    df = df[df.bed_rooms <= 10]
    df = df[df.bath_rooms <= 10]
    df = df[df.taxvaluedollarcnt <= df.taxvaluedollarcnt.quantile(0.75)]


    # split
    train, validate, test = split_data(df)
    
    return train, validate, test

    ###### split data #############

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    return train, validate, test

def model_sets(train,validate,test):
    '''
    Function drops the target of taxvaluedollarcnt column then splits data into 
    predicting variables (x) and target variable (y)
    ''' 

    x_train = train.drop(columns=['taxvaluedollarcnt'])
    y_train = train.taxvaluedollarcnt


    x_validate = validate.drop(columns=['taxvaluedollarcnt'])
    y_validate = validate.taxvaluedollarcnt

    x_test = test.drop(columns=['taxvaluedollarcnt'])
    y_test = test.taxvaluedollarcnt

    return x_train, y_train, x_validate, y_validate, x_test, y_test