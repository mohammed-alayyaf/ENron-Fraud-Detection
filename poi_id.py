#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import numpy as np
import math
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.grid_search import GridSearchCV
from tester import test_classifier
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')
from sklearn import tree
sys.path.append("../tools/")


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Data Exploration

#Sample data for one of the people in the dataset "Jeffrey Skilling"
print data_dict["SKILLING JEFFREY K"]

#Number of people in the datasets

print "Total number of people in the dataset: " + str(len(data_dict))

#Number of features in the dataset

print "Total number of features in the dataset: " + str(len(data_dict["SKILLING JEFFREY K"]))

#Number of POIs in the dataset

poi_count = 0
for i in data_dict:
    if data_dict[i]['poi']==True:
        poi_count=poi_count+1
print 'Number of poi in the dataset:', poi_count


### Task 2: Remove outliers

# first, we will check for outliers in the data by plotting it!
features = ["bonus", "salary"]
data = featureFormat(data_dict, features)
print(data.max())

for point in data:
    bonus = point[0]
    salary = point[1]
    plt.scatter(bonus, salary)

plt.xlabel("bonus")
plt.ylabel("salary")
plt.show()

# and as is it appears that we have one very imporant outlier, we just need to figure out which data point is it ?!

for i, v in data_dict.items():
    if v['salary'] != 'NaN' and v['salary'] > 10000000:
        print "Outliers: ", i

# and as it turned out that it was not a person !, but it was the "Total", so in this case removing the outlier
# would be the right move to make ..

### Next, we will remove the outlier

features = ["bonus","salary"]
data = featureFormat(data_dict, features)

data_dict.pop('TOTAL', 0)


# now let's take another look at the plot after removing the outlier


for point in data:
    bonus = point[0]
    salary = point[1]
    plt.scatter( bonus, salary )

plt.xlabel("bonus")
plt.ylabel("salary")
plt.show()

# and now  we have more normal plot, with the high data points being are people susbected for fraud "POI"


### Removing NaN values

# Load data into Pandas
df = pd.DataFrame.from_records(list(data_dict.values()))

# Convert to numpy nan
df.replace(to_replace='NaN', value=np.nan, inplace=True)
# df = df.replace(np.nan,'NaN', regex=True)

# DataFrame dimension
print df.shape
# print df.head()

# First, we will check for missing values and delete them, if found
print "Number of NaN's in each coulmn before cleaning: ", df.isnull().sum()

# it turns out we have huge number of NaN's value in each column
# next we will remove them to ensure data authenticity

df_imp = df.replace(to_replace=np.nan, value=0)
df_imp = df.fillna(0).copy(deep=True)
df_imp.columns = list(df.columns.values)
print "Number of NaN's in each coulmn after cleaning: ", df_imp.isnull().sum()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# now we will add new feature into our dataset, which is the bonus to salary ratio

for key in data_dict:
    ratio = float(data_dict[key]['bonus'])/float(data_dict[key]['salary'])
    if math.isnan(ratio):
        data_dict[key]['bonus_salary_ratio'] = 0
    else:
        data_dict[key]['bonus_salary_ratio'] = ratio

### Store to my_dataset for easy export below.
my_dataset = data_dict


if 'bonus_salary_ratio' not in features_list:
    features_list += ['bonus_salary_ratio']

# we added 'bonus salary ratio' as a new feature, next we will go through our feature selection process
# to decide which features to use


### Feature Selection

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


# for i in range(len(labels)):
#     labels[i] = int(labels[i])


# # labels and features print statements:
# print "Labels", labels

# for i in range(len(features)):
#     print "Features ", features[i]


selection=SelectKBest(k=14).fit(features,labels)
scores=selection.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs=list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
selection_best = dict(sorted_pairs[:14])
print selection_best

features_list = ['poi','salary', 'bonus', 'exercised_stock_options', 'total_stock_value', 'total_payments', 'loan_advances',
                'expenses', 'deferred_income', 'bonus_salary_ratio', 'long_term_incentive']


# Feature Scaling through MinMaxScale

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# Splitting the dataset into training and testing data

features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(features,labels, test_size=0.3,
                                                                                          random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# Naive Bayes

nb_clf = GaussianNB()
nb_clf.fit(features_train,labels_train)
pred = nb_clf.predict(features_test)

print("Naive Bayes Classifier: ")
print "Accuracy: " + str(accuracy_score(labels_test,pred))
print "Precision Score: " + str(precision_score(labels_test,pred))
print "Recall Score: " + str(recall_score(labels_test,pred))


# Decision Tree

dt_clf = tree.DecisionTreeClassifier()
dt_clf.fit(features_train,labels_train)
pred = dt_clf.predict(features_test)

print("Decision Tree Classifier: ")
print "Accuracy: " + str(accuracy_score(labels_test,pred))
print "Precision Score: " + str(precision_score(labels_test,pred))
print "Recall Score: " + str(recall_score(labels_test,pred))


# SVC

svc_clf = SVC()
svc_clf.fit(features_train,labels_train)
pred = svc_clf.predict(features_test)

print("SVC Classifier: ")
print "Accuracy: " + str(accuracy_score(labels_test,pred))
print "Precision Score: " + str(precision_score(labels_test,pred))
print "Recall Score: " + str(recall_score(labels_test,pred))


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


## Classifier Tuning using Grid Search

# Decision Tree
cv = cross_validation.StratifiedShuffleSplit(labels, n_iter=10)

parameters = {'max_depth': [1,2,3,4,5,6,7,8,9,10],'min_samples_split':[2,3,4,5], 'max_leaf_nodes':[2,3,4,5,6,7,8,9,10,15,20,30],
              'min_samples_leaf':[1,2,3,4,5,6,7,8], 'criterion':('gini', 'entropy'), 'splitter' : ('best','random')}



clf = tree.DecisionTreeClassifier()
clf = GridSearchCV(clf, parameters, cv=cv, scoring='f1')
clf = clf.fit(features, labels)
print clf.best_estimator_


# SVC

# parameters = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear','rbf'], 'gamma': [0.001, 0.0001]}]
#
# clf = SVC()
# clf = GridSearchCV(clf, parameters)
# clf = clf.fit(features, labels)
# print clf.best_estimator_


## Testing the best estimator against the test_classifier test set

test_classifier(clf.best_estimator_, my_dataset, features_list)

## Testing the Naive Bayes classifier against the test_classifier test set

test_classifier(nb_clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(nb_clf, my_dataset, features_list)
