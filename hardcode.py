import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class SVR(object):
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon
        
    def fit(self, X, y, epochs=100, learning_rate=0.1):
        self.sess = tf.Session()
        
        feature_len = X.shape[-1] if len(X.shape) > 1 else 1
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, feature_len))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        
        self.W = tf.Variable(tf.random_normal(shape=(feature_len, 1)))
        self.b = tf.Variable(tf.random_normal(shape=(1,)))
        
        self.y_pred = tf.matmul(self.X, self.W) + self.b
        
        #self.loss = tf.reduce_mean(tf.square(self.y - self.y_pred))
        #self.loss = tf.reduce_mean(tf.cond(self.y_pred - self.y < self.epsilon, lambda: 0, lambda: 1))
        
        # Second part of following equation, loss is a function of how much the error exceeds a defined value, epsilon
        # Error lower than epsilon = no penalty.
        self.loss = tf.norm(self.W)/2 + tf.reduce_mean(tf.maximum(0., tf.abs(self.y_pred - self.y) - self.epsilon))
#         self.loss = tf.reduce_mean(tf.maximum(0., tf.abs(self.y_pred - self.y) - self.epsilon))
        
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        opt_op = opt.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        
        for i in range(epochs):
            loss = self.sess.run(
                self.loss, 
                {
                    self.X: X,
                    self.y: y
                }
            )
            print("{}/{}: loss: {}".format(i + 1, epochs, loss))
            
            self.sess.run(
                opt_op, 
                {
                    self.X: X,
                    self.y: y
                }
            )
            
        return self
            
    def predict(self, X, y=None):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        y_pred = self.sess.run(
            self.y_pred, 
            {
                self.X: X 
            }
        )
        return y_pred
    
dataset = "D:/UNOM/3rd sem/ML_Lab/TataSteel.xlsx"
# x_train, y_train = load_data()
df = pd.read_excel(dataset)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Date'] = le.fit_transform(df['Date'])

from sklearn import preprocessing
mm = preprocessing.MinMaxScaler()
size = 600


x_train, y_train = preprocessing.normalize([df['Date'][:size],df['Close Price'][:size]], norm='l2', axis=1, copy=True, return_norm=False)


# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5]) 

# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])  

print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))


# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 
# plt.scatter(df['Date'], df['Close Price'], marker='x', c='r') 

# Set the title
plt.title("Tata Steel")
# Set the y-axis label
plt.ylabel('Price')
# Set the x-axis label
plt.xlabel('Days')
plt.show()

model = SVR(epsilon=0.2)

model.fit(x_train, y_train)



m = 2
c = 1

y = m * x_train + c
y += np.random.normal(size=(len(y),))
plt.plot(x_train, y, "x")

plt.plot(
    x_train, y, "x",
    x_train, model.predict(x_train), "-"
)
plt.legend(["actual", "prediction"])