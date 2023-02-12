# ML-Regression

You can learn K-NN Regression, Linear Regression, Polynomial Regression, Multiple Regression, Ridge and Lasso.

Unlike classification, regression predicts "value" of the target. 

You can think regression as "function".

---------------

## K-NN Regression

K-NN regression predicts target using the "average of several close values".

Here are the result of prediction using K-NN Regression

![Result image of K-NN Regression](/images/perch_50.jpg)

This model predicts 50cm tall perch's weight as 1000g which is similar to 40~45cm tall perches.

![Result image of K-NN Regression](/images/perch_100.jpg)

Can you tell the problem of K-NN Regression algorithm?

The problem is, when the model predicts target, it just uses given data.

And we have to find another algorithm which finds the graph(function). So that, we could know the general tendency of the data.

---------------------------------------

## linear Regression 

To solve the problem above, we need linear regression algorithm.

Linear regression finds the best parameter for linear function. 

Here is a general linear function : y = ax + b

'a' is called coefficient or weight. And 'b' is called intercept.

Both of them are called "model parameter."

Let's see the result of linear regression.

![Result of linear regression](/images/linear-regression.jpg)

It predicted 50cm tall perch's weight quite reasonably.

But, there's a problem. The graph can go through the X axis and then, target value could be a negative value. But, there is no negative value for weight!

So, we have to find another algorithm.

------------------------

## Polynomial Regression



