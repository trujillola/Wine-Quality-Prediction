
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# Set pandas options
pd.set_option('display.max_column',13)

# Load datastet
wines = pd.read_csv("./app/data/Wines.csv")


# Explore dataset :

# No missing values
# Min quality value = 3
# Max quality value = 8
# Seulement 6 vins ont obtenus une note de 3 !!

# print("Columns : \n" , wines.columns)
# print()
# print("Size of dataset : ",wines.shape)
# print()
# print("Head of dataset : \n",wines.head())
# print()
# print("Description of dataset : \n", wines.describe())
# print(wines["quality"].value_counts())



#############################
#   Preprocessing dataset   # 
#############################

# Drop Id column
wines.drop('Id', axis=1, inplace=True)

# delete duplicates
# X = wines.drop('quality', axis=1)
# print('Duplicated : ',wines.duplicated().sum())
# wines = wines.drop_duplicates(subset=X.columns, keep='first')
# print('Duplicated : ', wines.duplicated().sum())

# Get X and Y
X = wines.drop('quality', axis=1)
Y = wines['quality']

# Divide in train and test sets

X_train, X_test, Y_train ,Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

# Data augmentation
# Inspired by https://www.kaggle.com/code/bigironsphere/basic-data-augmentation-feature-reduction
# Substituting 50% of the features in each row with randomly sampled values from that feature's actual distribution.

def data_augmentation(train, n=1):
    a = np.arange(0, train.shape[1])
    train_aug = pd.DataFrame(index=train.index, columns=train.columns, dtype='float64')
    for i in tqdm(range(0, len(train))):
        #ratio of features to be randomly sampled
        AUG_FEATURE_RATIO = 0.2
        #to integer count
        AUG_FEATURE_COUNT = np.floor(train.shape[1]*AUG_FEATURE_RATIO).astype('int16')
    
        #randomly sample columns that will contain random values
        print(train.shape[1]-1)
        aug_feature_index = np.random.choice(train.shape[1]-2, AUG_FEATURE_COUNT, replace=False)
        aug_feature_index.sort()

        #obtain indices for features not in aug_feature_index
        feature_index = np.where(np.logical_not(np.in1d(a, aug_feature_index)))[0]      

        #first insert real values for features in feature_index
        train_aug.iloc[i, feature_index] = train.iloc[i, feature_index]

        #random row index to randomly sampled values for each features
        rand_row_index = np.random.choice(len(train), len(aug_feature_index), replace=True)
            
        #for each feature being randomly sampled, extract value from random row in train
        for n, j in enumerate(aug_feature_index):
            # Choose the index of a row where the quality is the same as the current row
            rows_same_quality = np.where(train['quality'] == train.iloc[i, -1])[0]
            index = np.random.choice(len(rows_same_quality))
            train_aug.iloc[i, j] = train.iloc[index, j]

    return train_aug

# train = pd.concat([X_train,Y_train], axis = 1)
# train_aug = data_augmentation(train,1)
# train_all = pd.concat([train, train_aug])

# print("train_aug : ", train_all.shape)

# X_train = train_all.drop('quality', axis=1)
# Y_train = train_all['quality']


###########################
#   Models construction   # 
###########################


# Train model RandomForest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve,StratifiedKFold, GridSearchCV,cross_val_score
from joblib import dump, load

if os.path.exists("/data/model_RandomForest.joblib") :
    print("Load model...")
    model = load("/data/model_RandomForest.joblib")
else :
    print("Train model...")
    cv = StratifiedKFold(4)
    param_grid = {"n_estimators": range(10, 16),"max_depth": range(5, 10), "min_samples_split": range(2, 6), "min_samples_leaf": range(1, 4)}
    grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv)
    grid.fit(X_train,Y_train)
    print(grid.best_score_)
    print(grid.best_params_)
    model = grid.best_estimator_
    dump(model, "/data/model_RandomForest.joblib")

print("Test score : ",model.score(X_test,Y_test))

# Train model LogisticRegression

from sklearn.linear_model import LogisticRegression

# Scale dataset 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
scaler = StandardScaler()
X = pd.DataFrame(    scaler.fit_transform(    X = wines.drop(['quality'], axis=1), y = wines['quality']  )       )


# polynomial_features= PolynomialFeatures(degree=2)

# X = polynomial_features.fit_transform(X)
X_train, X_test, Y_train ,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

if os.path.exists("/data/model_LogisticRegression.joblib") :
    print("Load model...")
    model = load("/data/model_LogisticRegression.joblib")
else : 
    print("Train model...")
    cv = StratifiedKFold(4)
    param_grid = {"C":np.logspace(-3,3,7), "penalty":["l2"], "solver":["liblinear","newton-cg","sag"]}
    grid = GridSearchCV(LogisticRegression(max_iter=10000), param_grid=param_grid, cv=cv)
    grid.fit(X_train,Y_train)
    print(grid.best_score_)
    print(grid.best_params_)
    model = grid.best_estimator_
    dump(model, "/data/model_LogisticRegression.joblib")

print("Test score : ",model.score(X_test,Y_test))


# Train model RidgeClassifier
from sklearn.linear_model import RidgeClassifier

if os.path.exists("/data/model_RidgeClassifier.joblib") :
    print("Load model...")
    model = load("/data/model_RidgeClassifier.joblib")
else : 
        print("Train model...")
        cv = StratifiedKFold(4)
        param_grid = {"alpha":[0.1, 1.0, 10.0], "solver":["svd","cholesky","lsqr","sparse_cg","sag","saga"]}
        grid = GridSearchCV(RidgeClassifier(max_iter=10000), param_grid=param_grid, cv=cv)
        grid.fit(X_train,Y_train)
        print(grid.best_score_)
        print(grid.best_params_)
        model = grid.best_estimator_
        dump(model, "/data/model_RidgeClassifier.joblib")

print("Test score : ",model.score(X_test,Y_test))


# Train model MLPClassifier

from sklearn.neural_network import MLPClassifier
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

if os.path.exists("/data/model_MLPC.joblib") :
    print("Load model...")
    model = load("/data/model_MLPC.joblib")
else : 
        print("Train model...")
        cv = StratifiedKFold(4)
        param_grid = {"hidden_layer_sizes": [(100,100), (100,100,100)],"alpha": [0.0001, 0.05]}
        grid = GridSearchCV(MLPClassifier(solver = "adam",learning_rate="adaptive", activation = "relu"), param_grid=param_grid, cv=cv)
        grid.fit(X_train,Y_train)
        print(grid.best_score_)
        print(grid.best_params_)
        model = grid.best_estimator_
        dump(model, "/data/model_MLPC.joblib")

print("Test score : ",model.score(X_test,Y_test))
