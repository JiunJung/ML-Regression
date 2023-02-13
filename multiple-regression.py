import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

# The more features you use, the higher model score will be.

# prepare input data (list of lists)
# df is short for data frame (pandas)
df = pd.read_csv('./perch_data.csv')
perch_full = df.to_numpy()

# prepare target data (not a list of lists)
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# Shuffle the data and split into train and test set.
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

# we use transformer to get more feature. It can create a lot of feature doing multiplication of each feature, square the feature, etc.
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)                      # you can think this code as a convention. And make sure that all of the preprocessing should use train input fitted set.
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input) # fit_transform does fit and transform simultaneously.
# print(train_poly.shape)              # (42, 9)
# print(poly.get_feature_names_out())  # ['x0' 'x1' 'x2' 'x0^2' 'x0 x1' 'x0 x2' 'x1^2' 'x1 x2' 'x2^2']
# print(test_poly.shape)

lr = LinearRegression()
lr.fit(train_poly, train_target)
# print(lr.score(train_poly, train_target)) # 0.9903183436982126
# print(lr.score(test_poly, test_target))   # 0.9714559911594125
# Now we solved the underfitted problem!

# Let's try multiple regression using more feature.
poly = PolynomialFeatures(degree=5, include_bias=False)
train_poly = poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
# print(train_poly.shape, test_poly.shape) # (42, 55) (14, 55) We have 55 features now.

lr.fit(train_poly, train_target)
# print(lr.score(train_poly, train_target)) # 0.9999999999997232
# print(lr.score(test_poly, test_target))   # -144.40564483377855
# train set score does improved! But, the model has been underfitted.

# So, we use "regularization" which means regulate some model parameter.
# Before that, we have to preprocess the feature.
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

ridge = Ridge()
ridge.fit(train_scaled, train_target)
# print(ridge.score(train_scaled, train_target)) # 0.9896101671037343
# print(ridge.score(test_scaled, test_target))   # 0.9790693977615379
# the model makes good score even it's feature number is 55.

# By the way, we can hadle "alpha" parameter to handle the degree of regularization. 
# (Parameters that we can change are called hyperparameters)
# Let's find the best alpha value.
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       ridge = Ridge(alpha=alpha)
       ridge.fit(train_scaled, train_target)
       train_score.append(ridge.score(train_scaled,train_target))
       test_score.append(ridge.score(test_scaled,test_target))

plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel("alpha")
plt.ylabel("R^2")
plt.show()

# According to the graph, 0.1 is the best alpha.

ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled,train_target)
# print(ridge.score(train_scaled,train_target)) # 0.9903815817570368
# print(ridge.score(test_scaled,test_target))   # 0.9827976465386983
# we've got the best test score.

# Now, let's do lasso now. The process is the same.

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       lasso = Lasso(alpha=alpha, max_iter=10000) #set max_iter to enough amount. Or error warning will be caused.
       lasso.fit(train_scaled,train_target)
       train_score.append(lasso.score(train_scaled, train_target))
       test_score.append(lasso.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel("alpha")
plt.ylabel("R^2")
plt.show()
# According to the graph, 10 is the best alpha

lasso = Lasso(alpha=10,max_iter=10000)
lasso.fit(train_scaled,train_target)
print(lasso.score(train_scaled,train_target)) # 0.9887624603020236
print(lasso.score(test_scaled,test_target)) # 0.9830309645308443

# Unlike Ridge, Lasso can even make coefficient to 0.
print(np.sum(lasso.coef_ == 0)) # 48
# lasso made 48 coefficient to 0.




