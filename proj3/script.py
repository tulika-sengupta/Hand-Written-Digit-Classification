
# coding: utf-8

# In[10]:

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import pickle


# In[11]:

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('C:/Users/divya/Google Drive/UB Courses/Spring \'17/CSE 574 Machine Learning/Assignments/PA3/mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


# In[12]:

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# In[4]:

def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    train_data_new = np.insert(train_data,0,1,axis=1)
    #print(initialWeights.shape)
    #print(train_data_new.shape[0])
    #print(train_data.shape[0])
    #print(train_data_new.shape[1])
    #print(train_data.shape[1])
    theta = sigmoid(np.dot(train_data_new,initialWeights))
   
    one_minus_theta = 1 - theta
    theta = np.reshape(theta, (n_data,1))
    log_theta = np.log(theta)
    #print(theta.shape)
    log_one_minus_theta = np.log(one_minus_theta)
    
    temp = np.dot(np.transpose(labeli), log_theta) + np.dot(np.transpose(1 - labeli), log_one_minus_theta)
    
    error = -temp/n_data
    
    error = error.flatten()
    
    #print(error)
    
    theta_minus_labeli = np.subtract(theta, labeli)
    
    #error_grad_temp = np.transpose(np.dot(theta_minus_labeli, train_data_new))
    
    #gc.collect()
    
    error_grad = (np.transpose(np.dot(np.transpose(theta_minus_labeli), train_data_new)))/n_data
    
    #print(error_grad.shape)
    error_grad = error_grad.flatten()
    
    #print(error_grad.shape)
    #error_grad = error_grad_temp/n_data
    
    
    

    return error, error_grad


# In[5]:

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    new_data = np.insert(data, 0, 1, axis=1)
    posterior = sigmoid (np.dot(new_data, W))
    #print(new_data.shape)
    #print(posterior.shape)
    label = np.argmax(posterior, axis=1)
    #print(label.shape)
    label = np.reshape(label, (data.shape[0],1))

    return label


# In[13]:

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
  
    
    
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
   
    w = np.reshape(params, (716,10))

    new_data = np.insert(train_data, 0, 1, axis=1)
    
    theta = np.exp(np.dot(new_data, w))
    theta_sum = np.reshape(np.sum(np.exp(np.dot(new_data, w)),axis=1),(n_data,1))
    theta_final= np.divide(theta, theta_sum)
    #print(theta_final.shape)
    error = - (np.sum((np.log(theta_final))*labeli))
    
   
    error_grad = np.dot(np.transpose(new_data),np.subtract(theta_final,labeli))
    
    error_grad = error_grad.flatten()
      
    
    #print(error)

    return error, error_grad


# In[14]:

def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    new_data = np.insert(data, 0, 1, axis=1)
        
        
    theta = np.exp(np.dot(new_data, W))
    theta_sum = np.reshape(np.sum(np.exp(np.dot(new_data, W)),axis=1),(data.shape[0],1))
    theta_final= np.divide(theta, theta_sum)
    
    
    label = np.argmax(theta_final, axis=1)
    #print(label.shape)
    label = np.reshape(label, (label.shape[0],1))
    

    return label


# In[8]:

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

f1 = open('params.pickle', 'wb') 
pickle.dump(W, f1) 
f1.close()

# In[ ]:

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
fit_label = train_label
fit_label = np.reshape(fit_label,(train_data.shape[0],))

print("Using linear kernel: \n")
f = SVC(kernel='linear')
f.fit(train_data,fit_label)
accuracy_1tr=f.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_1tr) + "\n")
accuracy_1va=f.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_1va) + "\n")
accuracy_1te=f.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_1te) + "\n")

print("Using radial basis function with value of gamma setting to 1: \n")
g = SVC(kernel='rbf',gamma=1.0)
g.fit(train_data,fit_label)
accuracy_2tr=g.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_2tr) + "\n")
accuracy_2va=g.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_2va) + "\n")
accuracy_2te=g.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_2te) + "\n")

print("Using radial basis function with value of gamma setting to default: \n")
h = SVC(kernel='rbf')
h.fit(train_data,fit_label)
accuracy_3tr=h.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_3tr) + "\n")
accuracy_3va=h.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_3va) + "\n")
accuracy_3te=h.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_3te) + "\n")

print("Using radial basis function with value of gamma setting to default and varying value of C \n")

print("for C=1")
i = SVC(kernel='rbf', C=1.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")


print("for C=10")
i = SVC(kernel='rbf', C=10.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")


print("for C=20")
i = SVC(kernel='rbf', C=20.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")


print("for C=30")
i = SVC(kernel='rbf', C=30.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")


print("for C=40")
i = SVC(kernel='rbf', C=40.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")


print("for C=50")
i = SVC(kernel='rbf', C=50.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")


print("for C=60")
i = SVC(kernel='rbf', C=60.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")


print("for C=70")
i = SVC(kernel='rbf', C=70.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")


print("for C=80")
i = SVC(kernel='rbf', C=80.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")


print("for C=90")
i = SVC(kernel='rbf', C=90.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")


print("for C=100")
i = SVC(kernel='rbf', C=100.0)
i.fit(train_data,fit_label)
accuracy_4tr=i.score(train_data,train_label)
print("Training dataset accuracy: "+ str(accuracy_4tr) + "\n")
accuracy_4va=i.score(validation_data,validation_label)
print("Validation dataset accuracy: "+ str(accuracy_4va) + "\n")
accuracy_4te=i.score(test_data,test_label)
print("Testing dataset accuracy: "+ str(accuracy_4te) + "\n")




# In[15]:


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()


# In[ ]:



