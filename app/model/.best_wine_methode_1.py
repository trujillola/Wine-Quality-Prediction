from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from scipy import optimize
import math
import os
from joblib import dump, load


# Get data
data_red = pd.read_csv("data/Wines.csv")
data_white = pd.read_csv("data/winequality-white.csv", sep=";")
data = data_red

data_for_pca = data.drop(['quality','Id'], axis=1)

# Scaling data
scaler = RobustScaler()
X_train_for_pca = pd.DataFrame(scaler.fit_transform(data_for_pca), columns=data_for_pca.columns)

# PCA to reduce dimension 5 --> 90% of original information kept in the 5 first components 
pca = PCA(n_components=8)
pca.fit(X_train_for_pca)
print("Explained variance of PCA : ",pca.explained_variance_ratio_.sum())

#Transform the data to the new basis
X_train_after_pca = pca.transform(data.drop(['quality','Id'], axis=1))
print(X_train_after_pca.shape)

# Computes polynomial features (x1^2, x2^2, x1*x2)
polynomial_features= PolynomialFeatures(degree=2)
X_train_poly = polynomial_features.fit_transform(X_train_after_pca)
print("Nb of potential variables : ",X_train_poly.shape)


if os.path.exists("model.joblib") :
    print("Load model...")
    lm = load("model.joblib")
else : 
    print("Train model...")
    
    # Perform linear regression -> Choose the variables with a cross validation ?
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
    hyper_params = [{'n_features_to_select': list(range(1, 46))}]
    lm = LinearRegression()
    lm.fit(X_train_poly,data['quality'])
    rfe = RFE(lm)
    model_cv = GridSearchCV(estimator = rfe, 
                            param_grid = hyper_params, 
                            scoring= 'r2', 
                            cv = folds, 
                            verbose = 1,
                            return_train_score=True)

    model_cv.fit(X_train_poly,data['quality'])

    cv_results = pd.DataFrame(model_cv.cv_results_)
    plt.figure(figsize=(16,6))

    plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
    plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
    plt.xlabel('number of features')
    plt.ylabel('r-squared')
    plt.title("Optimal Number of Features")
    plt.legend(['test score', 'train score'], loc='upper left')

    plt.show()
    # Choice of the optimal number of features 
    print(model_cv.best_params_)

n_features_optimal = 39

# Final Model with the optimal number of features
lm = LinearRegression()
lm.fit(X_train_poly,data['quality'])

rfe = RFE(lm, n_features_to_select=n_features_optimal)             
rfe = rfe.fit(X_train_poly,data['quality'])

    # Check the quality of the model  ----> BAD
predicted = lm.predict(X_train_poly)
truth = data['quality']
print("Coefficient of determination: %.2f" % r2_score(truth, predicted))


# Save the model
print("Save model...")
dump(lm, 'model.joblib')

# Check the parameters of the regression
# print('weights: ')
# print(lm.coef_)
# print('Intercept: ')
# print(lm.intercept_)

# Retrieve position of selected variables
rfe.ranking_[rfe.ranking_ > 1] = 0

X = X_train_poly[1].reshape(1,-1)
# print(rfe.ranking_)
# print(X)
# print(lm.coef_)
print(len(rfe.ranking_),X.shape[1],len(lm.coef_))
def function(X, coeff=lm.coef_, intercept=lm.intercept_ , ranks=rfe.ranking_):
    print(X.shape)
    result_list = []
    # Ici problème avec les X qui sont testés par la fonction d'optimisation : TypeError: 'numpy.float64' object is not iterable 
    for i,j,k in zip(ranks,X[0],coeff) :
        result_list.append(i * j * k)
    print(result_list)
    return - (  sum(result_list)  + intercept)

print("Result of function test on X :", function(X))

# # Find the minimum of the function
x0 = polynomial_features.fit_transform(pca.transform(pd.DataFrame(data = [[9.1,0.4,0.5,1.8,0.071,7.0,16.0,0.9946200000000001,3.21,0.69,12.5]],columns = data_for_pca.columns))).reshape(1,-1)
print(x0[0,44])
minimum = optimize.minimize(function, x0)
# xopt = minimum.x
# ymin = minimum.fun

# # print(xopt)
# # print(ymin)

# print(scaler.get_params())
# #User inverse_transform to revert the scaling and to go back to the original representation
# xopt_original = scaler.inverse_transform(pca.inverse_transform(xopt[1:3]).reshape(1,11))
# print(xopt_original)