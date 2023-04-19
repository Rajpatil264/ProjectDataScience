# importing libraries
import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt 
import seaborn as sn                   # For plotting graphs
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

# loading the data
train = pd.read_csv("E:/PROJECTS/DataScienceProj/train.csv")
test = pd.read_csv("E:/PROJECTS/DataScienceProj/test.csv")
print(train.columns)
print(test.columns)

print(train.shape, test.shape)
print(train.dtypes)
#printing first five rows of the dataset
print(train.head())

print(train['subscribed'].value_counts())

# Normalize can be set to True to print proportions instead of number 
print(train['subscribed'].value_counts(normalize=True))

# plotting the bar plot of frequencies
train['subscribed'].value_counts().plot.bar()
plt.show()

sn.distplot(train["age"])
plt.show()

train['job'].value_counts().plot.bar()
plt.show()

train['default'].value_counts().plot.bar()
plt.show()

print(pd.crosstab(train['job'],train['subscribed']))
job=pd.crosstab(train['job'],train['subscribed'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('Job')
plt.ylabel('Percentage')
plt.show()


print(pd.crosstab(train['default'],train['subscribed']))
default=pd.crosstab(train['default'],train['subscribed'])
default.div(default.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('default')
plt.ylabel('Percentage')
plt.show()


train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)

corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")
plt.show()

print(train.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

target = train['subscribed']
train = train.drop('subscribed',1)
# applying dummies on the train dataset
train = pd.get_dummies(train)
# splitting into train and validation with 20% data in validation set and 80% data in train set.
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=12)

# defining the logistic regression model
lreg = LogisticRegression()
# fitting the model on  X_train and y_train
lreg.fit(X_train,y_train)
# making prediction on the validation set
prediction = lreg.predict(X_val)
# calculating the accuracy score
print(accuracy_score(y_val, prediction))

# defining the decision tree model with depth of 4, you can tune it further to improve the accuracy score
clf = DecisionTreeClassifier(max_depth=4, random_state=0)
# making prediction on the validation set
predict = clf.predict(X_val)
# calculating the accuracy score
print(accuracy_score(y_val, predict))
test = pd.get_dummies(test)
test_prediction = clf.predict(test)

submission = pd.DataFrame()
# creating a Business_Sourced column and saving the predictions in it
submission['ID'] = test['ID']
submission['subscribed'] = test_prediction

# creating a Business_Sourced column and saving the predictions in it
submission['ID'] = test['ID']
submission['subscribed'] = test_prediction
submission.to_csv('submission.csv', header=True, index=False)