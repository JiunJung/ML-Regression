import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# We want to know weight of the perch by it's length. So we use regression algorithm.

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

# plt.scatter(perch_length,perch_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

# input should be a list of lists. While, target doen't need to be a list of lists.
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)
# print(train_input.shape, test_input.shape, train_target.shape, test_target.shape)

knr = KNeighborsRegressor()
# knr.fit(train_input,train_target)
# print(knr.score(train_input, train_target)) # coefficient of determination (R^2) : 0.9698 -> underfitting.
# print(knr.score(test_input, test_target)) # coefficient of determination (R^2) : 0.9928

# This model is underfitted. It happens when the algorithm is too simple. So, we have to make algorithm to be more complicated.
# So, we make k value (number of neighbors) smaller, so that the model can focus on small local area of the data.

knr.n_neighbors = 3

knr.fit(train_input, train_target)
print(knr.score(train_input,train_target)) # 0.9804
print(knr.score(test_input,test_target))   # 0.9746

# This model was trained well. Niether overfitted nor underfitted.

# Let's predict the weight of 50cm tall perch.
weight_of_50 = knr.predict([[50]])


distances, indexes = knr.kneighbors([[50]])

plt.scatter(train_input, train_target)
plt.scatter(50,weight_of_50 , marker='^')
plt.scatter(train_input[indexes,0],train_target[indexes],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# How about 100cm tall perch?
weight_of_100 = knr.predict([[100]])

distances, indexes = knr.kneighbors([[100]])

plt.scatter(train_input, train_target)
plt.scatter(100,weight_of_100 , marker='*')
plt.scatter(train_input[indexes,0],train_target[indexes],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
