# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:33:13 2019

@author: BY20064109
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix

os.chdir("D:/New folder/Hackathon/Hiring hackathon")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()
train.shape
test.shape

df_comb = pd.concat([train, test])

train.m13.value_counts(normalize = True) #highly imbalanced dataset
#0    0.99452
#1    0.00548

df_comb.rename(columns = {'m13':'target'}, inplace = True)

#checking for missing values
train.isnull().sum()
test.isnull().sum()

train['origination_date'] = pd.to_datetime(train['origination_date'], format = '%Y-%m-%d')
train['first_payment_date'] = pd.to_datetime(train['first_payment_date'], format = '%m/%Y')

test['origination_date'] = pd.to_datetime(test['origination_date'], format = '%d/%m/%y')
test['first_payment_date'] = pd.to_datetime(test['first_payment_date'], format = '%b-%y')

df_comb.source.value_counts(normalize = True)
df_comb.financial_institution.value_counts(normalize = True)

#train.pivot_table(index = 'source', columns = 'm13', agg_func = 'count')
pd.crosstab(df_comb.source,df_comb.target, normalize = 'index')
pd.crosstab(df_comb.financial_institution,df_comb.target, normalize = 'index')
pd.crosstab(df_comb.loan_purpose,df_comb.target, normalize = 'index')

#creating new column 'interest rate slabs'
df_comb['interest_rate_slabs'] = np.where(df_comb['interest_rate'] <= 3, 'le3',
     np.where((df_comb['interest_rate'] > 3) & (df_comb['interest_rate'] <= 3.5), '3-3.5',
              np.where((df_comb['interest_rate'] > 3.5) & (df_comb['interest_rate'] <= 4), '3.5-4',
                       np.where((df_comb['interest_rate'] > 4) & (df_comb['interest_rate'] <= 4.5), '4-4.5','greater4.5'))))

pd.crosstab(df_comb['interest_rate_slabs'], df_comb.target, normalize = 'index')
plt.bar(df_comb['interest_rate_slabs'], df_comb.target)
sns.countplot(x = 'interest_rate_slabs', hue = 'target', data = df_comb)
sns.boxplot(x = 'target', y = 'interest_rate', data = df_comb)
#loans with high interest rates are delinquent

pd.crosstab(df_comb.number_of_borrowers, df_comb.target, normalize = 'index')
#though there are more loan with 2 borrowers the delinquency rate is high with 1 borrower
sns.countplot(x = 'number_of_borrowers', hue = 'target', data = df_comb)

#it is a known fact that customers with low credit score are likely to be delinquent
sns.boxplot(x = 'target', y = 'borrower_credit_score', data = df_comb)

#though loan purpose A23 has high count, the delinquency rate is low compared to B12 and C86
pd.crosstab(df_comb.loan_purpose, df_comb.target, normalize = 'index')
sns.countplot(x = 'loan_purpose', hue = 'target', data = df_comb)

#delinquency is high when the insurance is paid by the lender
#though the count is only 5, the % is high in total compared to insurance paid by borrower 
pd.crosstab(df_comb.insurance_type, df_comb.target, normalize = 'index')

#m1 to m12 has values with delinquency in months.
#will create new variable m20 which sums up delinquency of all months after 
#changing the current values to either 0 or 1

df = df_comb
df['m1'] = np.where(df['m1'] == 0, 0, 1)
df['m2'] = np.where(df['m2'] == 0, 0, 1)
df['m3'] = np.where(df['m3'] == 0, 0, 1)
df['m4'] = np.where(df['m4'] == 0, 0, 1)
df['m5'] = np.where(df['m5'] == 0, 0, 1)
df['m6'] = np.where(df['m6'] == 0, 0, 1)
df['m7'] = np.where(df['m7'] == 0, 0, 1)
df['m8'] = np.where(df['m8'] == 0, 0, 1)
df['m9'] = np.where(df['m9'] == 0, 0, 1)
df['m10'] = np.where(df['m10'] == 0, 0, 1)
df['m11'] = np.where(df['m11'] == 0, 0, 1)
df['m12'] = np.where(df['m12'] == 0, 0, 1)

df['m20'] = df['m1']+df['m2']+df['m3']+df['m4']+df['m5']+df['m6']+df['m7']+df['m8']+df['m9']+df['m10']+df['m11']+df['m12']

#drop columns not required
df.drop(columns = ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12'], inplace = True)

#there are 19 financial institutions and few of them have very less values
#combining those with less values in to a new category and reducing the count
df.financial_institution.nunique()
df['financial_institution'] = np.where(df['financial_institution'] == 'OTHER', 'Other',
  np.where(df['financial_institution'] == 'Browning-Hart', 'Browning-Hart',
           np.where(df['financial_institution'] == 'Swanson, Newton and Miller', 'Swanson, Newton and Miller',
                    np.where(df['financial_institution'] == 'Edwards-Hoffman', 'Edwards-Hoffman',
                             np.where(df['financial_institution'] == 'Martinez, Duffy and Bird', 'Martinez, Duffy and Bird','Misc')))))

sns.heatmap(df.corr(), annot=True)
#no. of borrowers and co-borrower credit score are highly correlated
#we will drop co-borrower credit score

#drop columns not required
df.drop(columns = ['loan_id', 'co-borrower_credit_score', 'interest_rate','origination_date','first_payment_date'], inplace = True)

#one hot encoding for categorical variables
df = pd.get_dummies(df)

#create train and test datasets
Y_train = df.iloc[0:116058]['target']
X_train = df.drop('target', 1).iloc[0:116058]

X_test = df.drop('target', 1).iloc[116058:151924]

param_grid = {'eta':[0.05, 0.1, 0.15], 
              'max_depth':[6,7,8],
              'gamma': [0.5, 1, 1.5],
              'min_child_weight': [1, 5, 10]              
              }

xgb_model = xgb.XGBClassifier(n_estimators = 500,objective='binary:logistic',metric='auc',scale_pos_weight=2)
fold = StratifiedKFold(n_splits = 5, shuffle = True, random_state=2)
rs_cv = RandomizedSearchCV(xgb_model, param_grid, cv = fold.split(X_train, Y_train))

st=datetime.now()
rs_cv.fit(X_train, Y_train)
#end=datetime.now()
print("Time taken is:", datetime.now() - st)

best_params = rs_cv.best_params_
model_fit = xgb.XGBClassifier(params = best_params, n_estimators = 500,objective='binary:logistic',metric='auc',scale_pos_weight=2)
xgb_model = model_fit.fit(X_train, Y_train)

#prediction on test dataset
test_pred = xgb_model.predict(X_test)
Y_test = test_pred.astype(int)






