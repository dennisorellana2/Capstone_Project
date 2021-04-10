# -*- coding: utf-8 -*-
"""Data 670 Capstone Project Part 3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LtzE1yB4tQxfaJIq7cJQNYqTdiuHdeth

# Dennis Orellana
## Data 670 Capstone Project Part 3

## Purpose: This is part 3 of my Capstone Project. This section is comparing machine learning models for the "job_posting_final.csv".
"""

# Install the lazypredict library 
!pip install lazypredict

# Import Libraries
import pandas as pd 
import numpy as np 
import seaborn as sb
import matplotlib.pyplot as plt
import lazypredict


from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

# upload the csv file from local directory
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

# load the dataset from working directory 

job_posting_final = pd.read_csv('job_posting_final.csv')

"""## Split dataset into training and test"""

# set the X and y varaibles for Implementation of Logistic Regression
x = job_posting_final[['job_title', 'job_description', 'location', 'city', 'state', 'zip_code',
       'apply_url', 'company_name', 'companydescription', 'uniq_id',
       'crawl_timestamp', 'job_board', 'job_id', 'department', 'requirements',
       'benefits', 'telecommuting', 'has_company_logo', 'has_questions',
       'function', 'fraudulent', 'job_type', 'Full Time', 'I.T']].values

y = job_posting_final[['fraudulent']].values

# set the X and Y varaibles 
X = job_posting_final[['job_title', 'job_description', 'location', 'city', 'state', 'zip_code',
       'apply_url', 'company_name', 'companydescription', 'uniq_id',
       'crawl_timestamp', 'job_board', 'job_id', 'department', 'requirements',
       'benefits', 'telecommuting', 'has_company_logo', 'has_questions',
       'function', 'fraudulent', 'job_type', 'Full Time', 'I.T']].values

Y = job_posting_final[['fraudulent']].values

# Data split
from sklearn.model_selection  import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
         X, y, test_size=0.2, random_state=0)

# Standardizing the features:
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Data split shape
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

"""## Predictive Models##"""

# Builds the lazyclassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, Y_train, Y_test)

"""### Prints the model performance"""

# print the models
models

"""### Choosing three different classification algorithms from the list above which are

### Logistic Regression
### K Nearest Neighbors
###  Random Forest

# Logistic Regression
"""

# Training Model
from sklearn.linear_model import LogisticRegression
LgR = LogisticRegression(solver='liblinear')
LgR.fit(X_train, Y_train)

# Testing Model 
Y_pred = LgR.predict(X_test)
Y_test = Y_test.flatten()
Y_pred = Y_pred.flatten()
Y_test.shape, Y_pred.shape

# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(Y_pred, Y_test)

"""#### Find the Implementation of Logistic Regression"""

# Add a bias column to the X
X = np.c_[np.ones((x.shape[0], 1)), x]
X[:5]

# make y to match with the dimensions
y = y[:, np.newaxis]
y[:5]

# Create sigmoid function
def sigmoid(x, theta):
    z= np.dot(x, theta)
    return 1/(1+np.exp(-z))

# Create hypothesis function with the sigmoid
def hypothesis(theta, x):
    return sigmoid(x, theta)

# Create cost function using the formula
def cost_function(theta, x, y):
    m = X.shape[0]
    h = hypothesis(theta, x)
    return -(1/m)*np.sum(y*np.log(h) + (1-y)*np.log(1-h))

# Create the gradient descent function 
def gradient(theta, x, y):
    m = X.shape[0]
    h = hypothesis(theta, x)
    return (1/m) * np.dot(X.T, (h-y))

# Import the  optimization function
theta = np.zeros((X.shape[1], 1))
from scipy.optimize import minimize,fmin_tnc
def fit(x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(x, y.flatten()))
    return opt_weights[0]
parameters = fit(X, y, theta)

# calculate the final hypothesis
h = hypothesis(parameters, X)

# hypothesis outputs
def predict(h):
    h1 = []
    for i in h:
        if i>=0.5:
            h1.append(1)
        else:
            h1.append(0)
    return h1
y_pred = predict(h)

# Accuracy
accuracy = 0
for i in range(0, len(y_pred)):
    if y_pred[i] == y[i]:
        accuracy += 1
accuracy/len(y)

# Evaluation Logistic Regression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss
cm = confusion_matrix(Y_test, Y_pred)
acc = accuracy_score(Y_test, Y_pred)
F1 = f1_score(Y_test, Y_pred, average="micro") 

print(cm)
print(acc)
print(F1)

# classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

# Plot Confusion Matrix with Heatmap 

sb.heatmap(cm, annot=True, cmap='Blues')

# Plot Percentage Confusion Matrix with Heatmap
sb.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='winter')

"""# K Nearest Neighbors"""

# Training Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)

# Testing Model 
Y_pred = knn.predict(X_test)

# Accuracy Score
accuracy_score(Y_pred,Y_test)

from sklearn.model_selection import cross_val_score
# search for an optimal value of K for KNN
k_range = list(range(1, 10))

k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

"""The optimal value of K for KNN is 2."""

# Evaluation K Nearest Neighbors
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss
cm = confusion_matrix(Y_test, Y_pred)
acc = accuracy_score(Y_test, Y_pred)
F1 = f1_score(Y_test, Y_pred, average="micro") 

print(cm)
print(acc)
print(F1)

# classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

# Plot Confusion Matrix with Heatmap 

sb.heatmap(cm, annot=True, cmap='Greens')

# Plot Percentage Confusion Matrix with Heatmap
sb.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='summer')

"""# Random Forest """

# Training Model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_features=5,n_estimators=10)
rfc.fit(X_train, Y_train)

# Testing Model 
Y_pred = rfc.predict(X_test)
Y_test = Y_test.flatten()
Y_pred = Y_pred.flatten()

# Accuracy Score
accuracy_score(Y_pred,Y_test)

# Evaluation Random Forest 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss
cm = confusion_matrix(Y_test, Y_pred)
acc = accuracy_score(Y_test, Y_pred)
F1 = f1_score(Y_test, Y_pred, average="micro") 

print(cm)
print(acc)
print(F1)

# classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

# Plot Confusion Matrix with Heatmap 

sb.heatmap(cm, annot=True, cmap='Purples')

# Plot Percentage Confusion Matrix with Heatmap
sb.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='cool')

"""## 10-Fold K-Fold Cross Validation"""

# Logistic regression model performance using cross_val_score
Log_score = cross_val_score(LogisticRegression(solver='liblinear'), X, Y,cv=10)
print(Log_score)

# Average logistic regression cross_val_score
Log_score.mean()

# K Nearest Neighbors model performance using cross_val_score
knn_score = cross_val_score(KNeighborsClassifier(n_neighbors=2), X, Y,cv=10)
print(knn_score)

# Average K Nearest Neighbors cross_val_score
knn_score.mean()

# Random Forest model performance using cross_val_score
rf_score = cross_val_score(RandomForestClassifier(n_estimators=10),X, Y,cv=10)
print(rf_score)

# Average Random Forest cross_val_score
rf_score.mean()

"""## Computing AUROC and ROC curve values"""

# import AUROC and ROC curve
from sklearn.metrics import roc_curve, roc_auc_score

# Prediction probabilities
r_probs = [0 for _ in range(len(Y_test))]

lr_probs = LgR.predict_proba(X_test)
rf_probs = rfc.predict_proba(X_test)

lr_probs = lr_probs[:, 1]
rf_probs = rf_probs[:, 1]

# Calculate AUROC
r_auc = roc_auc_score(Y_test, r_probs)
lr_auc = roc_auc_score(Y_test, lr_probs)
rf_auc = roc_auc_score(Y_test, rf_probs)

# Print AUROC scores
print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Random Forest: AUROC = %.3f' % (rf_auc))
print('Logistic Regression: AUROC = %.3f' % (lr_auc))

# Calculate ROC curve

r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)

"""
##Plot the ROC curve"""

# Plot ROC Curve
plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression (AUROC = %0.3f)' % lr_auc)

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()

"""The best overall models are Logistic Regression and Random Forest.

#Resources

•	https://github.com/codebasics/py/blob/master/ML/12_KFold_Cross_Validation/12_k_fold.ipynb

•	https://github.com/justmarkham/scikit-learn-videos/blob/master/07_cross_validation.ipynb

•	https://www.kaggle.com/shivanirana63/fake-job-prediction-ensemble-modeling

•	https://github.com/Suji04/Contraceptive-Method-prediction/blob/master/classifier.py

•	https://github.com/dataprofessor/code/blob/master/python/ROC_curve.ipynb

•	https://regenerativetoday.com/logistic-regression-with-python-using-an-optimization-function/
"""