
# coding: utf-8

# In[93]:


import numpy as np
import random
import pylab
import time


# $1.$ Implement in python the ridge regression with gradient descent. We will call this algorithm regression_gradient. Note that we now have parameters w and b we want to learn on the training set, as well an hyper-parameter to control the capacity of our model: $\lambda$. There are also hyper-parameters for the optimization: the step-size $\eta$, and potentially the number of steps.

# In[120]:


class RidgeRegression:
    def __init__(self):
        """
        Constructeur de la classe. Prend les paramètres données à la
        constuction de la classe et initialise ses attribues.

        Parameters
        ----------
        mu : float
            Taux d'apprentissage
        """
        
    ''' 
    def plot_function(self, train_data, title):
        plt.figure()
        x1 = np.linspace(-10, 10, 100)
        y1 = self.theta[1:]*x + self.theta[0]
        yy = sin(x1)+0.3*x1 -1
        
        plt.plot(X, Y, '.')
        plt.plot(x1, yy, '-')
        plt.plot(x1, y1, c='r', lw=2, label='y = w*x + b')        
        plt.grid()
        plt.legend(loc='lower right')
        plt.title(title)
        plt.show()
    '''   

    def regression_gradient(self, train_data, lam, eta, max_iter):
        #np.random.seed(10)
        #theta = np.random.rand(data_set.shape[1]) 
    
        iteration = 0
        n = train_data.shape[0]
        d = train_data.shape[1]
        t = train_data[:, -1]
    
        self.theta = np.ones(d)
        print(self.theta)
    
        x0 = np.ones(n)
        X0 = np.reshape(x0, (n, 1))
        X1 = train_data[:,:-1]
        X = np.c_[X0, X1]

        while iteration < max_iter:
            #self.plot_function(train_data, 'Iteration no: ' + str(iteration))
            self.theta = (1-2*lam*eta)*self.theta + 2*eta*(np.dot((t-np.dot(X, self.theta)),X))
            iteration += 1
        
        print("b = ", self.theta[0])
        print("w = ", self.theta[1:])
        self.bias = self.theta[0]
        self.weight = self.theta[1:]
        return self.weight, self.bias


# $2.$ Consider the function h(x) = sin(x) + 0.3x − 1. Draw a dataset $D_n$ of pairs (x, h(x)) with n = 15 points where x is drawn uniformly at random in the interval [−5, 5]. Make sure to use the same set $D_n$ for all the plots below.

# In[121]:


X = np.random.uniform(-5,5,15)
Y = sin(X) + 0.3*(X) - 1

XX = np.linspace(-10, 10, 50)
YY = sin(XX)+0.3*XX -1

data = np.vstack((X, Y))
data_set = np.transpose(data)


# $3.$ With $\lambda$ = 0, train your model on $D_n$ with the algorithm regression_gradient(). Then plot on the interval [−10, 10]: the points from the training set $D_n$, the curve h(x), and the curve of the function learned by your model using gradient descent. Make a clean legend. 
# Remark: The solution you found with gradient descent should converge to the straight line that is closer from the n points (and also to the analytical solution). Be ready to adjust your step-size (small enough) and number of iterations (large enough) to reach this result.

# In[147]:


# regression_gradient(train_data, lam, eta, max_iter):
model1 = RidgeRegression()
w1, b1 = model1.regression_gradient(data_set, 0, 0.001, 10)
print(w1, b1)

w2, b2 = model1.regression_gradient(data_set, 0, 0.0001, 1000)
print(w2, b2)

w3, b3 = model1.regression_gradient(data_set, 0, 0.000001, 1000000)
print(w3, b3)

plt.figure(figsize=(10,6))
x1 = np.linspace(-10, 10, 100)
y1 = w1*XX + b1
y2 = w2*XX + b2
y3 = w3*XX + b3
yy = sin(x1)+0.3*x1 -1
        
plt.plot(X, Y, '.')
plt.plot(XX, YY, '-')
plt.plot(XX, y1, c='r', lw=2, label='y = w*x + b')      
plt.plot(XX, y2) 
plt.plot(XX, y3)
plt.grid()
plt.legend(('training point', 'y=sin(x)+0.3x-1','λ=0, η=0.000001, max_iter=10','λ=0, η=0.000001, max_iter=1000','λ=0, η=0.000001, max_iter=1000000'))
plt.show()


# $4.$ on the same graph, add the predictions you get for intermediate value of $\lambda$, and for a large value of $\lambda$. Your plot should include the value of $\lambda$ in the legend. It should illustrate qualitatively what happens when $\lambda$ increases.

# In[149]:


# regression_gradient(train_data, lam, eta, max_iter):
model1 = RidgeRegression()
w1, b1 = model1.regression_gradient(data_set, 0, 0.000001, 10)
print(w1, b1)

w2, b2 = model1.regression_gradient(data_set, 0, 0.000001, 1000)
print(w2, b2)

w3, b3 = model1.regression_gradient(data_set, 0, 0.000001, 1000000)
print(w3, b3)

w4, b4 = model1.regression_gradient(data_set, 10, 0.0001, 1000)
print(w4, b4)

w5, b5 = model1.regression_gradient(data_set, 100, 0.0001, 1000)
print(w5, b5)

w6, b6 = model1.regression_gradient(data_set, 0.000000001, 0.0001, 1000)
print(w6, b6)

plt.figure(figsize=(10,6))
#x1 = np.linspace(-10, 10, 100)
y1 = w1*XX + b1
y2 = w2*XX + b2
y3 = w3*XX + b3
y4 = w4*XX + b4
y5 = w5*XX + b5
y6 = w6*XX +b6
#yy = sin(x1)+0.3*x1 -1
        
plt.plot(X, Y, '.')
plt.plot(XX, YY, '-')
plt.plot(XX, y1, c='r', lw=2, label='y = w*x + b')      
plt.plot(XX, y2) 
plt.plot(XX, y3)
plt.plot(XX, y4)
plt.plot(XX, y5)
plt.plot(XX, y6)
plt.grid()
plt.legend(('training point', 'y=sin(x)+0.3x-1','λ=0, η=0.000001, max_iter=10','λ=0, η=0.000001, max_iter=1000','λ=0, η=0.000001, max_iter=1000000',
           'λ=10, η=0.0001, max_iter=1000','λ=100, η=0.0001, max_iter=1000', 'λ=0.000000001, η=0.0001, max_iter=1000'))
plt.show()


# $5.$ Draw another dataset $D_{test}$ of 100 points by following the same procedure as $D_n$. Train your linear model on $D_n$ for $\lambda$ taking values in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]. For each value of $\lambda$, measure the
# average quadratic loss on $D_{test}$. Report these values on a graph with $\lambda$ on the x-axis and the loss value on the y-axis.

# In[165]:


# regression_gradient(train_data, lam, eta, max_iter):

#X = np.random.uniform(-5,5,15)
#Y = sin(X) + 0.3*(X) - 1

XX1 = np.linspace(-10, 10, 100)
YY1 = sin(XX1)+0.3*XX1-1

model2 = RidgeRegression()
w7, b7 = model2.regression_gradient(data_set, 0.0001, 0.001, 1000)
y7 = 1/(2*100)*np.sum(((w7*XX1+b7)- YY1)**2)
print(y7)

w8, b8 = model2.regression_gradient(data_set, 0.001, 0.001, 1000)
y8 = 1/(2*100)*np.sum(((w8*XX1+b8)- YY1)**2)

w9, b9 = model2.regression_gradient(data_set, 0.01, 0.001, 1000)
y9 = 1/(2*100)*np.sum(((w9*XX1+b9)- YY1)**2)

w10, b10 = model2.regression_gradient(data_set, 0.1, 0.001, 1000)
y10 = 1/(2*100)*np.sum(((w10*XX1+b10)- YY1)**2)

w11, b11 = model2.regression_gradient(data_set, 1, 0.001, 1000)
y11 = 1/(2*100)*np.sum(((w11*XX1+b11)- YY1)**2)

w12, b12 = model2.regression_gradient(data_set, 10, 0.001, 1000)
y12 = 1/(2*100)*np.sum(((w12*XX1+b12)- YY1)**2)

w13, b13 = model2.regression_gradient(data_set, 100, 0.001, 1000)
y13 = 1/(2*100)*np.sum(((w13*XX1+b13)- YY1)**2)

plt.figure(figsize=(10,6))

lamb = (0.0001, 0.001, 0.01, 0.1, 1, 10, 100)
los = (y7, y8, y9, y10, y11, y12, y13)
print(los)
plt.plot(lamb, los, '-')

plt.grid()
plt.show()


# $6.$ Use the technique studied in problem 1.3 above to learn a non-linear function of x. Specifically, use Ridge regression with the fixed preprocessing $\phi_{poly^l}$ described above to get a polynomial regression of order l. Apply this technique with $\lambda$ = 0.01 and different values of l. Plot a graph similar to question 2.2 with all the prediction functions you got. Don’t plot too many functions to keep it readable and precise the value of l in the legend.
