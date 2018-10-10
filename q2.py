
# coding: utf-8

# In[134]:


get_ipython().magic('pylab inline')
import numpy as np
import random
import pylab
import time


# $1.$ Implement in python the ridge regression with gradient descent. We will call this algorithm regression_gradient. Note that we now have parameters w and b we want to learn on the training set, as well an hyper-parameter to control the capacity of our model: $\lambda$. There are also hyper-parameters for the optimization: the step-size $\eta$, and potentially the number of steps.

# In[135]:


def regression_gradient(train_data, lam, eta, max_iter):
    np.random.seed(10)
    weight = np.random.rand(data_set.shape[1]-1) 
    bias = -1
    
    iteration = 0
    n = train_data.shape[0]
    d = train_data.shape[1] - 1
    t = train_data[:, -1]

    while iteration < max_iter:
        for i in range(0,d):
            xs = data_set[:,i]
            xx = np.reshape(xs,(n,1))
            #weight = (1-2*eta*lam)*weight - 2*eta*(train_data[:,i])*((np.sum(train_data[:,i])*weight-np.sum(Y)))
            weight = (1-2*eta*lam)*weight + 2*eta*(np.dot((t-np.dot(xx, weight)),xx))
        iteration += 1
        
    print("w = ", weight)
    return weight, bias


# $2.$ Consider the function h(x) = sin(x) + 0.3x − 1. Draw a dataset $D_n$ of pairs (x, h(x)) with n = 15 points where x is drawn uniformly at random in the interval [−5, 5]. Make sure to use the same set $D_n$ for all the plots below.

# In[136]:


X = np.random.uniform(-5,5,15)
Y = sin(X) + 0.3*(X) - 1

XX = np.linspace(-10, 10, 50)
YY = sin(XX)+0.3*XX -1

data = np.vstack((X, Y))
data_set = np.transpose(data)


# $3.$ With $\lambda$ = 0, train your model on $D_n$ with the algorithm regression_gradient(). Then plot on the interval [−10, 10]: the points from the training set $D_n$, the curve h(x), and the curve of the function learned by your model using gradient descent. Make a clean legend. 
# Remark: The solution you found with gradient descent should converge to the straight line that is closer from the n points (and also to the analytical solution). Be ready to adjust your step-size (small enough) and number of iterations (large enough) to reach this result.

# In[137]:


# regression_gradient(train_data, lam, eta, max_iter):
w1,b1 = regression_gradient(data_set, 0, 0.00001, 100)
y1 = w1*X + b1

w2,b2 = regression_gradient(data_set, 0, 0.00001, 1000)
y2 = w2*X + b2

w3,b3 = regression_gradient(data_set, 0, 0.00001, 100000)
y3 = w3*X + b3

plt.plot(X, Y, '.')
plt.plot(XX, YY, '-')
plt.plot(X, y1, '-')
plt.plot(X, y2, '-')
plt.plot(X, y3, '-')
plt.legend(('training point','y=sin(x)+0.3x-1','λ=0, η=0.001, max_iter=100','λ=0, η=0.001, max_iter=1000','λ=0, η=0.001, max_iter=100000'))
plt.show()


# $4.$ on the same graph, add the predictions you get for intermediate value of $\lambda$, and for a large value of $\lambda$. Your plot should include the value of $\lambda$ in the legend. It should illustrate qualitatively what happens when $\lambda$ increases.

# In[138]:


# regression_gradient(train_data, lam, eta, max_iter):
w1,b1 = regression_gradient(data_set, 0, 0.00001, 100)
y1 = w1*X + b1

w2,b2 = regression_gradient(data_set, 0, 0.00001, 1000)
y2 = w2*X + b2

w3,b3 = regression_gradient(data_set, 0, 0.00001, 100000)
y3 = w3*X + b3

w4,b4 = regression_gradient(data_set, 10, 0.001, 1000)
y4 = w4*X+b4

w5,b5 = regression_gradient(data_set, 100, 0.001, 1000)
y5 = w5*X+b5

plt.figure(figsize=(10,6))
plt.plot(X, Y, '.')
plt.plot(XX, YY, '-')
plt.plot(X, y1, '-')
plt.plot(X, y2, '-')
plt.plot(X, y3, '-')
plt.plot(X, y4, '-')
plt.plot(X, y5, '-')
plt.legend(('training point','y=sin(x)+0.3x-1', 'λ=0, η=0.00001, max_iter=100', 'λ=0, η=0.00001, max_iter=1000', 'λ=0, η=0.00001, max_iter=100000', 
            'λ=10, η=0.001, max_iter=1000', 'λ=100, η=0.001, max_iter=1000'))
plt.show()


# $5.$ Draw another dataset $D_{test}$ of 100 points by following the same procedure as $D_n$. Train your linear model on $D_n$ for $\lambda$ taking values in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]. For each value of $\lambda$, measure the
# average quadratic loss on $D_{test}$. Report these values on a graph with $\lambda$ on the x-axis and the loss value on the y-axis.

# In[145]:


X_test = np.random.uniform(-5,5,100)
L = sin(X_test) + 0.3*(X_test) - 1
print(L)

wt1, bt1 = regression_gradient(data_set, 0.0001, 0.001, 1000)
Z1 = wt1*(X_test)+bt1
print(Z1)
#Z1 = (wt1*X_test+bt1-L)**2

wt2, bt2 = regression_gradient(data_set, 0.001, 0.001, 1000)
Z2 = (wt2*X_test+bt2-L)**2

wt3, bt3 = regression_gradient(data_set, 0.01, 0.001, 1000)
Z3 = (wt3*X_test+bt3-L)**2

wt4, bt4 = regression_gradient(data_set, 0.1, 0.001, 1000)
Z4 = (wt4*X_test+bt4-L)**2

wt5, bt5 = regression_gradient(data_set, 1, 0.001, 1000)
Z5 = (wt5*X_test+bt5-L)**2

wt6, bt6 = regression_gradient(data_set, 10, 0.001, 1000)
Z6 = (wt6*X_test+bt6-L)**2

wt7, bt7 = regression_gradient(data_set, 100, 0.001, 1000)
Z7 = (wt7*X_test+bt7-L)**2

plt.figure(figsize=(10,6))
plt.plot(X, Y, '.')
plt.plot(XX, YY, '-')
#plt.plot(X_test, Z1, '-')
#plt.plot(X_test, Z2, '-')
#plt.plot(X_test, Z3, '-')
#plt.plot(X_test, Z4, '-')
#plt.plot(X_test, Z5, '-')
#plt.plot(X_test, Z6, '-')
#plt.plot(X_test, Z7, '-')
plt.show()

