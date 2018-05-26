# -*- coding: utf-8 -*-
"""
Created on Sat May 12 10:24:03 2018

@author: Shravan
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url)

print(data.head())
#so convert to csv using ';' as delimiter

data = pd.read_csv(dataset_url , sep = ';')
print(data.head())
print(data.shape)


print(data.describe())

#take target as wine quality
y = np.array(data.quality)
X = np.array(data.drop('quality' , axis=1)  )


X_train , X_test , y_train , y_test = train_test_split(X , y ,random_state = 123, test_size = 0.2 ,stratify = y )
#it's good practice to stratify your sample by the target variable. 
#this will ensure your training set looks similar to your test set, making your evaluation metrics more reliable.

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.mean(axis=0))
print(X_test_scaled.std(axis=0))





from sklearn.ensemble import RandomForestRegressor
#import cross-validation pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#shortcut for standardization
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

#model_parameters such as how to classify or create branches will be decided(learn) by models using mean_squared_error ..
#hyper_parameters such as no. of decision trees .. should be set by users 

print(pipeline.get_params())

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
#use __ when tuning model through pipeline

#Split your data into k equal parts, or "folds" (typically k=10).
#Preprocess k-1 training folds.
#Train your model on the same k-1 folds.
#Preprocess the hold-out fold using the same transformations from step (2).
#Evaluate your model on the same hold-out fold.
#Perform steps (2) - (5) k times, each time holding out a different fold.
#Aggregate the performance across all k folds. This is your performance metric.
#FOR THIS WE HAVE SCIKIT GRIDSEARCHCV WHICH DOES CREATE NO. OF FOLDS AND TEST ON DIFF HYPERPARAMETERS 
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
#gridsearch is permutations of all hyperparameters
clf.fit(X_train , y_train)
print(clf.best_params_)
# default __max_depth is best acc to o/p :o

# clf refits entire dataset automatically if clf.refit is true
print(clf.refit)
#so our model is set to predict




#evaluate performance
from sklearn.metrics import mean_squared_error , r2_score , confusion_matrix , accuracy_score

y_pred = clf.predict(X_test)
print(r2_score(y_test , y_pred))
#print(confusion_matrix(y_test , y_pred))
print(mean_squared_error(y_test , np.round(y_pred)))
print(accuracy_score(y_test , np.round(y_pred)))
print(confusion_matrix(y_test , np.round(y_pred)))

#joblib (or pickle) to persist our model
from sklearn.externals import joblib
joblib.dump(clf, 'C:/Users/Shravan/Desktop/wine_rf_regressor.pkl')


#clf2 = joblib.load('rf_regressor.pkl')
#clf2.predict(X_test)
 



