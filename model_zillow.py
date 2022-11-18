import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TweedieRegressor
########################## Modeling & Evaluate Functions ##############################
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

def simple_sqft_model(train,validate):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the simple linear regression model and computes the predictions 
    and then adds them to the dataframe  
    ''' 
    x_train = train[['finished_sqft']]
    y_train = train.taxvaluedollarcnt

    x_validate = validate[['finished_sqft']]
    y_validate = validate.taxvaluedollarcnt

    lm_model = LinearRegression().fit(x_train, y_train)
    train['lm_predictions'] = lm_model.predict(x_train)

    validate['lm_predictions'] = lm_model.predict(x_validate)

    return train, validate

def simple_bath_model(train,validate):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the simple linear regression model and computes the predictions 
    and then adds them to the dataframe  
    ''' 
    x_train = train[['finished_sqft']]
    y_train = train.taxvaluedollarcnt

    x_validate = validate[['finished_sqft']]
    y_validate = validate.taxvaluedollarcnt

    lm_model = LinearRegression().fit(x_train, y_train)
    train['lm_predictions'] = lm_model.predict(x_train)

    validate['lm_predictions'] = lm_model.predict(x_validate)

    return train, validate



def regression_errors(y, yhat):
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    ESS = ((yhat - y.mean())**2).sum()
    TSS = ESS + SSE
    
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    baseline = np.repeat(y.mean(), len(y))
    
    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    return SSE, MSE, RMSE
    
def better_than_baseline(y, yhat):
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    
    if SSE < SSE_baseline:
        print('My OSL model performs better than baseline')
    else:
        print('My OSL model performs worse than baseline. :( )')