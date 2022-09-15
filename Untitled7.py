#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#toyotaCoralla_df = pd.read_csv('ToyotaCorlla.csv').iloc[:1000,:]
#toyotaCorolla_df = toyotaCorlla_df.rename(columns={'Age_08_04': 'Age', 'Quarterly_Tax': 'Tax'})
#predictors = ['Age', 'KM', 'Fuel_Type', 'HP', 'Met_Color', 'Automatic', 'CC', 'Doors', 'Tax', 'Weight']
predictors = ['Spending', 'Address_is_res', 'Gender=male', 'Web order', '1st_update_days_ago', 'last_update_days_ago', 'Freq', 'source_w', 'source_x']
outcome = 'Price'

X = x_train_pur[predictors]
y = y_train

# user grid search to find optimized tree
param_grid = {
    'max_depth': [5, 10, 15, 20, 25],
    'min_impurity_decrease': [0, 0.001, 0.005, 0.01],
    'min_samples_split': [10,20,30,40,50]
}

gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(x_train_pur, y_train_pur)
print('Initial parameters: ', gridSearch.best_params_)

param_grid = {
    'max_depth': [3,4,5,6,7,8,9,10,11,12],
    'min_impurity_decrease': [0,0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.008],
    'min_samples_split': [14, 15, 16, 18, 20, ]
}

gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Improved parameters: ', gridSearch.best_params_)

regTree = gridSearch.best_estimator_

regressionSummary(train_y,regTree.predict(train_X))
regressionSummary(valid_y, regTree.predict(valid_X))


# In[ ]:




