import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#  اول حاجه هنجيب الdata و هنئراها باستخدام panda.read_csv البتئرا اي CSV فيل
# the data is separated by semicolons so it is hard to read for humans so we will remove the semicolon and place spaces
# this data has the shape (1599,12) and contains twelve features for each example like acidity, citric acid, chlorides, etc
# what we want is figure out the quality of the wine
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

# Split data into training and test sets
# The test set will be used to find the best hyper-parameters
# the training set will be used to train the AI algorithm
# we will make the test set 20% of the total data
# Y is the output
# X is the training data examples
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# Declare data pre-processing steps
# in this step we are standardizing the data.
# meaning we want the data to be in a more regular form.
# If the data is in a more regular form like a circle
# and our target is to reach its center then it will be a lot easier than if the data was in a wierd elongated shape.
# the following function makes the same standardization to the test set as the training set
# so as the test set remains a proper representative to the training set
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# Declare hyper-parameters to tune
# hyper-parameters are like learning rate, the depth of a nueral network, the number of nurons in each layer,
# and there are also hyper-parameters for optimization methods like adam optimization.
# we are trying to find the best hyper-parameters from the test set to use on the tranning set
# the following is a set of hyper-parameters the we will choose from using grid search
hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

# we will use the grid search to find the best hyper-parameters
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# we will fit these hyper-parameters to the training set
clf.fit(X_train, y_train)

# Evaluate model pipeline on test data
pred = clf.predict(X_test)
print(r2_score(y_test, pred))
print(mean_squared_error(y_test, pred))

# Save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')