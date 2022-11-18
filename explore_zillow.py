import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression

#################### Statistical and Visuals Functions ##################################
alpha = 0.05 
def get_stats_sqft(train):
    '''
    Function gets results of pearsonsr statistical test for finished_sqft and taxvaluedollarcnt
    '''

    r, p = stats.pearsonr(train.finished_sqft, train.taxvaluedollarcnt)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'pearsonsr test = {r:.4f}')
    

def get_stats_bath(train):
    '''
    Function gets results of pearsonsr statistical test for bath_rooms and taxvaluedollarcnt
    '''

    r, p = stats.pearsonr(train.bath_rooms, train.taxvaluedollarcnt)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'pearsonsr test = {r:.4f}')

def get_stats_built(train):
    '''
    Function gets results of pearsonsr statistical test for year_built and taxvaluedollarcnt
    '''

    r, p = stats.pearsonr(train.year_built, train.taxvaluedollarcnt)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'pearsonsr test = {r:.4f}')

def get_stats_bed(train):
    '''
    Function gets results of pearsonsr statistical test for bed_rooms and taxvaluedollarcnt
    '''

    r, p = stats.pearsonr(train.bed_rooms, train.taxvaluedollarcnt)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'pearsonsr test = {r:.4f}')


def select_kbest(x_train, y_train):
    '''
    Function gets results of select kbest test for our train data
    '''
    kbest = SelectKBest(f_regression, k=2)
    _ = kbest.fit(x_train, y_train)
    kbest_results = pd.DataFrame(
        dict(p=kbest.pvalues_, f=kbest.scores_),
                                 index = x_train.columns)
    return kbest_results

    

