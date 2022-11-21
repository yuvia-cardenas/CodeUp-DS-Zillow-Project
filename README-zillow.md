# CodeUp-DS-Zillow-Project
 
### Project Goals

* Discover the key drivers of property value for single family properties.

* Use features to develop a machine learning model that predicts the property value for single family properties.

#### My initial hypothesis is that drivers of property value will be specific or combined elements/features that either cause an outright increase/decrease of property value.

### The Plan
* Aquire data from the CodeUp database
* Prepare data for exploration by creating tailored columns from existing data
#### Explore data in search of drivers by asking the basic following questions:
* What is average value of a single family home?
* What features should we investigate?
#### Develop a Model to predict property value
* Use drivers identified to build predictive models of different types
* Evaluate models on train and validate data samples
* Select the best model based on RSME
* Evaluate the best model on test data samples
#### Draw conclusions

### Steps to Reproduce
* Clone this repo.
* Confirm variables from user env.py file as
        user = 'your user name', 
        pwd = 'your password', 
        host = 'data.codeup.com'password pwd, etc.)
* Acquire the data from CodeUp database
* Put the data in the file containing the cloned repo.
* Run notebook
### Conclusions

* Lasso Lars Regression model RMSE scores:

        * 1.564690e+10 on training data samples
        * 2.385735e+10 on validate data samples
        * 1.542134e+10 on test data samples

### Key Takeaway

#### Lasso Lars model out performed baseline model on train data set and the test data set. 

### Recommendations

   * Consider eliminating outliers through and through  
   * Consider deeper dive investigation on bedrooms
   * Use bedrooms feature in future modeling
   * Consider the strategy to predict on transformed bathroom feature
       to account for half bath computation