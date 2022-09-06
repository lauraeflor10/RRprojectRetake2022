#!/usr/bin/env python
# coding: utf-8

# # Reproductible Research
# 
# ## BREAST CANCER PREDICTION

# ## TEAM MEMBER
# 
# 1. Laura Erika Rozalia Florencia 
# 2. Orkhan Amrullayev 
# 3. Srinesh Heshan Fernando 

# # PROJECT DESCRIPTION

# We would like to classified the cancer diagnosis, whether it's benign or malignant based on some observations. In this code, we are trying to replicate what's on paper that we proposed to the Professor previously. What's our project:
# 
# 1. On paper, the author are using kNN (0.99 accuracy), SVM (0.96 accuracy), Logistic Regression (0.97 accuracy) and Naive Bayes (0.95 accuracy)
# 2. We are trying to replicate only SVM and Logistic Regression in this code, because we got the same accuracy, thus we assume that we will get the same number for other model
# 3. In our code, we are using Logistic Regression (97% accuracy), SVM (96% accuracy) and Decision Tree (94% accuracy)
# 4. There is a diffence when we are using Decision Tree  
# 
# Ten real-valued features are computed for each cell nucleus (information from the website):
# 
# a) radius (mean of distances from center to points on the perimeter)   
# b) texture (standard deviation of gray-scale values)  
# c) perimeter  
# d) area  
# e) smoothness (local variation in radius lengths)  
# f) compactness (perimeter^2 / area - 1.0)  
# g) concavity (severity of concave portions of the contour)  
# h) concave points (number of concave portions of the contour)  
# i) symmetry  
# j) fractal dimension ("coastline approximation" - 1)  
#         
# The datasets are linear and separable using all 30 input features, with 569 number of instances.
# 
# About the class, the class distribution and target are 212 malignant and 357 benign.

# # IMPORTING LIBRARY & DATA

# In[2]:


# Firstly we should import the data for data manipulation using dataframes (pandas), for statistical analysis (numpy), 
# for data visualization (matplotlib) and statistical data visualization (seaborn)

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Import breast cancer data using sklearn

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() 
#cancer : Get the data target


# In[4]:


# Get the component from the data which we get from the library

cancer.keys()


# In[5]:


# Get the description from sklearn library

print(cancer['DESCR'])


# In[6]:


# The classification of the cancer

print(cancer['target_names'])


# In[7]:


# Get the target for observation, both 0 and 1 is the interpretation of the target_names 

print(cancer['target']) 


# In[8]:


# Check what data that we can get from the library source file

print(cancer['feature_names'])


# In[9]:


print(cancer['data'])


# In[10]:


# Check the data dimension in the dataset
# 569 rows with 30 columns

cancer['data'].shape


# In[11]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

# Get the first 5 data on dataset
df_cancer.head()


# In[12]:


# Get the last 5 data on dataset
df_cancer.tail()


# # DATA VISUALIZATION

# In[13]:


# Check the data visualization using seaborn pairplot. We do the observation for each 'mean' from the target in the data 
# observation.
# We can see that the data distribute clearly into 2 clusters and we have the combination observation for each parameter

# Orange color = benign
# Blue color = malignant

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )


# In[14]:


# We get the data comparison from the target, for which one can be categorized in 'Malignant' or 'Benign'.
# And here we can see that the patient with 'Benign' status is way more high than 'Malignant'

sns.countplot(df_cancer['target'], label = "Count") 


# In[15]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[16]:


# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter and get the heatmap

plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 


# # FIND THE SOLUTION IN THE TRAINING MODEL (1)

# In[17]:


# Let's drop the target label coloumns and here we see the original dataset

X = df_cancer.drop(['target'],axis=1)
X


# In[18]:


# Output result from 'Target', the result is the diagnosis/ classification
# on patients
y = df_cancer['target']
y


# In[19]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)


# In[20]:


X_train.shape


# In[21]:


X_test.shape


# In[22]:


y_train.shape


# In[23]:


y_test.shape


# ## FIRST TRIAL SVM ALGORITHMS
# 
# We train the model that we build previously. 
# In the first trial, we will use Kernel Support Vector Machine (SVM).

# In[24]:


# SVC = C-Support Vector Classification 

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)


# ## EVALUATING THE MODEL

# In[25]:


# Now we can do the classification for the model. The final product is the heatmap of the classified target.
# We do it step by step and here is the heatmap for the original dataset

y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)


# In[26]:


sns.heatmap(cm, annot=True)


# In[27]:


print(classification_report(y_test, y_predict))


# ## IMPROVING THE MODEL

# In[28]:


# We get the minimum and maximum values from each parameter and get the scale for new result from data observation
min_train = X_train.min()
min_train


# In[29]:


range_train = (X_train - min_train).max()
range_train


# In[30]:


X_train_scaled = (X_train - min_train)/range_train
X_train_scaled


# In[31]:


sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)


# In[32]:


sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


# In[33]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# ## SECOND TRIAL SVM ALGORITHMS
# 
# We do another trial for the improved data and here we will get the better accuration.

# In[34]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[35]:


# Here we already get the heatmap for the new dataset based on the final observation
y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")


# In[36]:


print(classification_report(y_test, y_predict))


# # FIND THE SOLUTION IN THE TRAINING MODEL (2)
# 
# We are using Logistic Regression for our second test. 

# In[37]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[38]:


y_pred = classifier.predict(X_test)
y_pred


# In[39]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")


# In[40]:


print(classification_report(y_test, y_pred))


# # FIND THE SOLUTION IN THE TRAINING MODEL (3)
# 
# The last, now we check the result from Decision Tree method.

# In[41]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[42]:


y_pred3 = classifier.predict(X_test)
y_pred3


# In[43]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred3)
sns.heatmap(cm, annot=True, fmt="d")


# In[44]:


print(classification_report(y_test, y_pred3))


# # SUMMARY
# 
# We are using 3 different algorithms to test our prediction. Those 3 gave different accuracy to classify the model for breast cancer patient. Here is the result:
# 1. Kernel SVM: 96%
# 2. Logistic Regression: 97% 
# 3. Decision Tree: 94%  
# 
# So for this case, the best algorithms to be used in the test is the Logistic Regression 

# # SOURCES:
# 1. https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# 2. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html?highlight=breast#sklearn.datasets.load_breast_cancer
# https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_self_training_varying_threshold.html#sphx-glr-auto-examples-semi-supervised-plot-self-training-varying-threshold-py
# 3. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html 
# 4. https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
# 5. https://chrisalbon.com/code/machine_learning/support_vector_machines/svc_parameters_using_rbf_kernel/#:~:text=gamma%20is%20a%20parameter%20of,decision%20region%20is%20very%20broad.
# 6. https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989
# 7. https://intellipaat.com/community/19362/what-is-the-difference-between-svc-and-svm-in-scikit-learn
# 8. https://intellipaat.com/blog/tutorial/machine-learning-tutorial/svm-algorithm-in-python/
# 9. https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3
# 10. https://www.dataquest.io/blog/sci-kit-learn-tutorial/#:~:text=Scikit%2Dlearn%20is%20a%20free,libraries%20like%20NumPy%20and%20SciPy%20.
# 

# ### Cheers!
