import numpy as np
  
import pandas as pd
  
from sklearn.model_selection import train_test_split
  
import matplotlib.pyplot as plt
  
# Lasso Regression
  
class LassoRegression() :
      
    def __init__( self, learning_rate, iterations, l1_penality ) :
          
        self.learning_rate = learning_rate
          
        self.iterations = iterations
          
        self.l1_penality = l1_penality
          
    # Function for model training
              
    def fit( self, X, Y ) :
          
        # no_of_training_examples, no_of_features
          
        self.m, self.n = X.shape
          
        # weight initialization
          
        self.W = np.zeros( self.n )
          
        self.b = 0
          
        self.X = X
          
        self.Y = Y
          
        # gradient descent learning
                  
        for i in range( self.iterations ) :
              
            self.update_weights()
              
        return self
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :
             
        Y_pred = self.predict( self.X )
          
        # calculate gradients  
          
        dW = np.zeros( self.n )
          
        for j in range( self.n ) :
              
            if self.W[j] > 0 :
                  
                dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) ) 
                           
                         + self.l1_penality ) / self.m
          
            else :
                  
                dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) ) 
                           
                         - self.l1_penality ) / self.m
  
       
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
          
        # update weights
      
        self.W = self.W - self.learning_rate * dW
      
        self.b = self.b - self.learning_rate * db
          
        return self
      
    # Hypothetical function  h( x ) 
      
    def predict( self, X ) :
      
        return X.dot( self.W ) + self.b
      
  
def main() :
      
    # Importing dataset
      
    dataset = "D:/UNOM/3rd sem/ML_Lab/TataSteel.xlsx"
    # x_train, y_train = load_data()
    df = pd.read_excel(dataset)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    df['Date'] = le.fit_transform(df['Date'])

    from sklearn import preprocessing
    mm = preprocessing.MinMaxScaler()
    size = 600


    X = df[['Date', 'Open Price', 'High Price', 'Low Price']]
    y = df['Close Price']
    # Model training
      
    model = LassoRegression( iterations = 1000, learning_rate = 0.01, l1_penality = 500 )
  
    model.fit(X, y)
      
    # Prediction on test set
  
    Y_pred = model.predict(X)
      
    print( "Predicted values ", np.round(Y_pred)) 
      
    print("Real values", y)
      
    print( "Trained W        ", round(model.W[0], 2 ) )
      
    print( "Trained b        ", round( model.b, 2 ) )
      
    # Visualization on test set 
      
    plt.scatter( X, y, color = 'blue' )
      
    plt.plot( X, Y_pred, color = 'orange' )
      
    plt.title( 'Stock Prediction' )
      
    plt.xlabel( 'Time' )
      
    plt.ylabel( 'Revenue' )
      
    plt.show()
      
  
if __name__ == "__main__" : 
      
    main()

