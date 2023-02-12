import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Perch data. Feature : length, Target : weight
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

# Make instance of Linear Regression model.
lr = LinearRegression()

lr.fit(train_input, train_target)

# print(lr.predict([[50]])) # result : 1241 -> looks reasonable than using K-NN regression.

# y = ax + b -> we can say "a" as a coefficient or weight and "b" as a intercept. Both of them are "model parameter".
# trainig ML model means findinf the best model parameters.
# print(lr.coef_, lr.intercept_) # result : [39.01714496] -709.0186449535474 

plt.scatter(train_input, train_target)
plt.plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_])
plt.scatter(50,lr.predict([[50]]), marker='D')
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# Let's see the score of this model.
print(lr.score(train_input, train_target)) # 0.9398463339976041
print(lr.score(test_input, test_target))   # 0.824750312331356
# It looks overfitted. but, actually the score of train set is not good enough. So, We can't say that it is overfitted.
# Instead, we can say that it is underfitted. Because both of the scores are not good enough.\
# Plus, when you look at the graph of linear function, you could recognize that the 'y' value can be negative value. But there are no negative value for weight!
# So, We will use another regression algorithm called "polynomial regression".
