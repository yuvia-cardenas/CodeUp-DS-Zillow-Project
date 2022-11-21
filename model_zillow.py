import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TweedieRegressor

########################## Modeling Functions ###########################

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

def predictions(train):
    '''
    Function takes the baseline and creates dataframe
    '''
    predictions = pd.DataFrame({
    'Actual': train.taxvaluedollarcnt})
    return predictions.head()

def simple_lm_model(train,validate):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the simple linear regression model and computes the predictions 
    and then adds them to the dataframe  
    ''' 
    x_train = train[['finished_sqft','bath_rooms','year_built']]
    y_train = train.taxvaluedollarcnt

    x_validate = validate[['finished_sqft','bath_rooms','year_built']]
    y_validate = validate.taxvaluedollarcnt

    lm_model = LinearRegression().fit(x_train, y_train)
    train['lm_predictions'] = lm_model.predict(x_train)
    validate['lm_predictions'] = lm_model.predict(x_validate)

    lm_co = lm_model.coef_
    lm_int = lm_model.intercept_

    simp_co = pd.Series(lm_model.coef_, index=x_train.columns).sort_values()
    print(simp_co)
    

def lasso_model(train,validate):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the lasso lars regression model and computes the predictions 
    and then adds them to the dataframe  
    ''' 
    x_train = train[['finished_sqft','bath_rooms','year_built']]
    y_train = train.taxvaluedollarcnt

    x_validate = validate[['finished_sqft','bath_rooms','year_built']]
    y_validate = validate.taxvaluedollarcnt

    lars = LassoLars(alpha=1).fit(x_train, y_train)
    train['lars_predictions'] = lars.predict(x_train)
    validate['lars_predictions'] = lars.predict(x_validate)

    lars_co = pd.Series(lars.coef_, index=x_train.columns).sort_values()
    print(lars_co)

def glm_model(train,validate):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the generalized linear regression model and computes the predictions 
    and then adds them to the dataframe  
    ''' 
    x_train = train[['finished_sqft','bath_rooms','year_built']]
    y_train = train.taxvaluedollarcnt

    x_validate = validate[['finished_sqft','bath_rooms','year_built']]
    y_validate = validate.taxvaluedollarcnt

    glm_model = TweedieRegressor(power=0, alpha=1).fit(x_train, y_train)
    train['glm_predictions'] = glm_model.predict(x_train)
    validate['glm_predictions'] = glm_model.predict(x_validate)

    glm_co = pd.Series(glm_model.coef_, index=x_train.columns).sort_values()
    print(glm_co)

############################# Evaluate the Models on Train #########################
def baseline_mean_errors(train):
    y = train.taxvaluedollarcnt
    baseline = np.repeat(y.mean(), len(y))
    
    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
      "\nRMSE: ", round(RMSE, 2))

def lm_errors(train):
    y = train.taxvaluedollarcnt
    yhat = train.lm_predictions

    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
      "\nRMSE: ", round(RMSE, 2))
    

def lars_errors(train):
    y = train.taxvaluedollarcnt
    yhat = train.lars_predictions
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
      "\nRMSE: ", round(RMSE, 2))


def glm_errors(train):
    y = train.taxvaluedollarcnt
    yhat = train.glm_predictions
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
      "\nRMSE: ", round(RMSE, 2))
