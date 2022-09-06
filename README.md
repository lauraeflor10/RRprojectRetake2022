We would like to classified the cancer diagnosis, whether it's benign or malignant based on some observations. In this code, we are trying to do the replication, which we are using the different data. Just for a reminder about our project:

On paper, the author are using kNN (0.99 accuracy), SVM (0.96 accuracy), Logistic Regression (0.97 accuracy) and Naive Bayes (0.95 accuracy)
We are trying to replicate only SVM and Logistic Regression in this code, because we got the same accuracy, thus we assume that we will get the same number for other model
In our code, we are using Logistic Regression (97% accuracy), SVM (96% accuracy) and Decision Tree (94% accuracy)
There is a diffence when we are using Decision Tree
Real data consist of 768 rows, and the replication has 286 rows
In this dataset, we practically change the float or text value to number as provided in the next part
==== Ten real-valued features are computed for each cell nucleus (information from the website):

a) age: 10-19 (1), 20-29 (2), 30-39 (3), 40-49 (4), 50-59 (5), 60-69 (6), 70-79 (7), 80-89 (8), 90-99 (9).
b) menopause: lt40, ge40, premeno.
c) tumor-size: 0-4 (4), 5-9 (9), 10-14 (14), 15-19 (19), 20-24 (24), 25-29 (29), 30-34 (34), 35-39 (39), 40-44 (44), 45-49 (49), 50-54 (54), 55-59 (59).
d) inv-nodes: 0-2 (2), 3-5 (5), 6-8 (8), 9-11 (11), 12-14 (14), 15-17 (17), 18-20 (20), 21-23 (23), 24-26 (26), 27-29 (29), 30-32 (32), 33-35 (35), 36-39 (39).
e) node-caps: yes (1), no (0).
f) deg-malig: 1, 2, 3.
g) breast: left (0), right (1).
h) breast-quad: left-up, left-low, right-up, right-low, central.
i) irradiat: 1, 0 (the part where it's malignant = 1 or benign = 0 in the previous replication)
j) class: no-recurrence-events, recurrence-events

Class Distribution:
no-recurrence-events: 201 instances
recurrence-events: 85 instances
