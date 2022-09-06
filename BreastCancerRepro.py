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

# We would like to classified the cancer diagnosis, whether it's benign or malignant based on some observations. In this code, we are trying to do the replication, which we are using the different data. Just for a reminder about our project:
# 
# 1. On paper, the author are using kNN (0.99 accuracy), SVM (0.96 accuracy), Logistic Regression (0.97 accuracy) and Naive Bayes (0.95 accuracy)
# 2. We are trying to replicate only SVM and Logistic Regression in this code, because we got the same accuracy, thus we assume that we will get the same number for other model
# 3. In our code, we are using Logistic Regression (97% accuracy), SVM (96% accuracy) and Decision Tree (94% accuracy)
# 4. There is a diffence when we are using Decision Tree  
# 5. Real data consist of 768 rows, and the replication has 286 rows  
# 6. In this dataset, we practically change the float or text value to number as provided in the next part
# 
# ====
# Ten real-valued features are computed for each cell nucleus (information from the website):
# 
# a) age: 10-19 (1), 20-29 (2), 30-39 (3), 40-49 (4), 50-59 (5), 60-69 (6), 70-79 (7), 80-89 (8), 90-99 (9).  
# b) menopause: lt40, ge40, premeno.  
# c) tumor-size: 0-4 (4), 5-9 (9), 10-14 (14), 15-19 (19), 20-24 (24), 25-29 (29), 30-34 (34), 35-39 (39), 40-44 (44), 45-49 (49), 50-54 (54), 55-59 (59).  
# d) inv-nodes: 0-2 (2), 3-5 (5), 6-8 (8), 9-11 (11), 12-14 (14), 15-17 (17), 18-20 (20), 21-23 (23), 24-26 (26), 27-29 (29), 30-32 (32), 33-35 (35), 36-39 (39).  
# e) node-caps: yes (1), no (0).  
# f) deg-malig: 1, 2, 3.  
# g) breast: left (0), right (1).  
# h) breast-quad: left-up, left-low, right-up, right-low, central.  
# i) irradiat: 1, 0 (the part where it's malignant = 1 or benign = 0 in the previous replication)  
# j) class: no-recurrence-events, recurrence-events  
# 
# Class Distribution:  
# no-recurrence-events: 201 instances  
# recurrence-events: 85 instances  

# # IMPORTING LIBRARY & DATA

# In[1]:


# Firstly we should import the data for data manipulation using dataframes (pandas), for statistical analysis (numpy), 
# for data visualization (matplotlib) and statistical data visualization (seaborn)

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[105]:


# Import breast cancer from CSV

data = pd.read_csv("breast_cancer_repro.csv")
data


# In[12]:


data.keys()


# In[13]:


# Get the target for observation, both 0 and 1 is the interpretation of the target_names 

print(data['irradiat']) 


# In[14]:


# Check the data dimension in the dataset
# 286 rows with 10 columns

data.shape


# In[60]:


# Get the first 5 data on dataset
data.head()


# In[61]:


# Get the last 5 data on dataset
data.tail()


# # DATA VISUALIZATION

# In[106]:


# Check the data visualization using seaborn pairplot. We do the observation for each 'mean' from the target in the data 
# observation.
# We can see that the data distribute clearly into 2 clusters and we have the combination observation for each parameter

# Orange color = benign
# Blue color = malignant

sns.pairplot(data, hue = 'irradiat', vars = ['deg-malig', 'node-caps'])


# In[107]:


# We get the data comparison from the target, for which one can be categorized in 'Malignant' or 'Benign'.
# And here we can see that the patient with 'Benign' status is way more high than 'Malignant'

sns.countplot(data['irradiat'], label = "Count") 


# In[108]:


# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter and get the heatmap

plt.figure(figsize=(20,10)) 
sns.heatmap(data.corr(), annot=True) 


# # FIND THE SOLUTION IN THE TRAINING MODEL (1)

# In[109]:


# Let's drop the target label coloumns which contain text

X = data.drop(['irradiat', 'menopause', 'breast-quad', 'Class'],axis=1)
X


# In[110]:


# Output result from 'Target' or 'Irradiat', the result is the diagnosis/ classification
# on patients
y = data['irradiat']
y


# In[111]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)


# In[112]:


X_train.shape


# In[113]:


X_test.shape


# In[114]:


y_train.shape


# In[115]:


y_test.shape


# ## FIRST TRIAL SVM ALGORITHMS
# 
# We train the model that we build previously. 
# In the first trial, we will use Kernel Support Vector Machine (SVM).

# In[116]:


# SVC = C-Support Vector Classification 

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)


# ## EVALUATING THE MODEL

# In[117]:


# Now we can do the classification for the model. The final product is the heatmap of the classified target.
# We do it step by step and here is the heatmap for the original dataset

y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)


# In[118]:


sns.heatmap(cm, annot=True)


# In[119]:


print(classification_report(y_test, y_predict))


# ## IMPROVING THE MODEL

# In[120]:


# We get the minimum and maximum values from each parameter and get the scale for new result from data observation
min_train = X_train.min()
min_train


# In[121]:


range_train = (X_train - min_train).max()
range_train


# In[122]:


X_train_scaled = (X_train - min_train)/range_train
X_train_scaled


# In[132]:


# Here we want to see the spread of the malignant degree and tumor size
# but it seems like the result is not like we are expected

sns.scatterplot(x = X_train['deg-malig'], y = X_train['tumor-size'], hue = y_train)


# In[133]:


sns.scatterplot(x = X_train_scaled['deg-malig'], y = X_train_scaled['tumor-size'], hue = y_train)


# In[134]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# ## SECOND TRIAL SVM ALGORITHMS
# 
# We do another trial for the improved data and here we will get the better accuration.

# In[135]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[136]:


# Here we already get the heatmap for the new dataset based on the final observation
y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")


# In[137]:


print(classification_report(y_test, y_predict))


# # FIND THE SOLUTION IN THE TRAINING MODEL (2)
# 
# We are using Logistic Regression for our second test. 

# In[138]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[139]:


y_pred = classifier.predict(X_test)
y_pred


# In[140]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")


# In[141]:


print(classification_report(y_test, y_pred))


# # FIND THE SOLUTION IN THE TRAINING MODEL (3)
# 
# The last, now we check the result from Decision Tree method.

# In[142]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[143]:


y_pred3 = classifier.predict(X_test)
y_pred3


# In[144]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred3)
sns.heatmap(cm, annot=True, fmt="d")


# In[145]:


print(classification_report(y_test, y_pred3))


# # SUMMARY
# 
# We are using 3 different algorithms to test our prediction. Those 3 gave different accuracy to classify the model for breast cancer patient. Here is the result:
# 1. Kernel SVM: 74%
# 2. Logistic Regression: 81% 
# 3. Decision Tree: 74%  
# 
# So for this case, the best algorithms to be used in the test is the Logistic Regression. The best algorithms is actually consistent with the first observation. However, the accuracy value are different, as shown below:  
# Observation || SVM || LR  || Decision Tree  
# First       || 96% || 97% || 94%  
# Second      || 74% || 81% || 74%  
# 
# The degree of accuracy are different, and in our second data, we have some limitation because the data is less than the initial one. 

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
# 11. https://datahub.io/machine-learning/breast-cancer#readme

# ### Cheers!
