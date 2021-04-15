## Introduction

Classification and regression are two types of supervised machine learning. In classification algorithms, it is a form in which you can train a model to use the features that calculates the probability of the observed case belonging to each of a number of possible classes and predicting an appropriate label. There are binary classification and multiclasss classification.

Now we try some classification models in Azure.

## 1. Binary classification

For the outcome variable, as called as the label, if it has two classes and we want to predict one of these two, the classification model that we represent would be a binary classifier, which could predict whether or not the label is "true" or "false".

Now take a look at a patient data set, our target is to predict whether or not a patient should be tested for diabetes based on some medical data.

upload the data set now:

```python
import pandas as pd
diabetes = pd.read.csv('data/diabetes.csv')
diabetes.head()
```

The information that this data set contains includes Pregancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, Seruminsulin, BMI, DiabetesPedigree, Age and Diabetic. The feature Diabetic is the label that we perform prediction, the value 0 for patients who tested negative for diabetes and 1 for patients who tested positive. Other columns(most) are the explanotory variables to predict the label.

![image](https://user-images.githubusercontent.com/71245576/114948047-47742900-9e1c-11eb-9c80-0c6b58d0fbb2.png)

Separate features and labels now.

```python
# Separate features and labels
features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
label = 'Diabetic'
X, y = diabetes[features].values, diabetes[label].values

for n in range(0,4):
    print("Patient", str(n+1), "\n  Features:",list(X[n]), "\n  Label:", y[n])
```

compare the feature distributions for each label value(each class of the label):
```python
from matplotlib import pyplot as plt
%matplotlib inline

features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
for col in features:
    diabetes.boxplot(column=col, by='Diabetic', figsize=(6,6))
    plt.title(col)
plt.show()
```

We can notice that Pregancies and Age show markedly different distributions for diabetic patients than for non-diabetic patients. See Pregancie V.S. Diabetic

![image](https://user-images.githubusercontent.com/71245576/114948401-e731b700-9e1c-11eb-9195-5b92ea772da8.png)

Let's split the data to training data set and testing data set.

```python
from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))
```
Train the binary classification model, the model that we choose is the logistic regression model, it is very widely used in machine learning areas. Regularization is a functionality that help punish some features for overfitting. We will discuss it in another article.

```python
# Train the model
from sklearn.linear_model import LogisticRegression

# Set regularization rate
reg = 0.01

# train a logistic regression model on the training set
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)
print (model)
```
Now let's see the predicted labels and actual labels and comparison of them.

```python
predictions = model.predict(X_test)
print('Predicted labels: ', predictions)
print('Actual labels:    ' ,y_test)
```

It seems not good, but we only got the values from 6 of the total set.

![image](https://user-images.githubusercontent.com/71245576/114948893-ee0cf980-9e1d-11eb-9861-91eebedfd25a.png)

test the accuracy of the model now

```python
from sklearn.metrics import accuracy_score

print('Accuracy: ', accuracy_score(y_test, predictions))
```

Woof! the accuracy is very okay, the accuracy is 0.7894. We should know the accuracy is just a metric about if or not the prediction matches the value of the actual label, there must be some other metrics to evaluate the performance! especially when the labeling makes the data is very skew, consider precision and recall accordingly!

```python
from sklearn. metrics import classification_report

print(classification_report(y_test, predictions))
```

The precision means that the proportion of correction that the prediction of the model made. The recall is to describe out of all of the instances of the class in the test dataset, how many did the model identify and F1 is an average metric takes both precision and recall into account. Let's see the result:

![image](https://user-images.githubusercontent.com/71245576/114949292-b3f02780-9e1e-11eb-8dae-399c85c91298.png)

Let's see the overall precision and recall:
```python
from sklearn.metrics import precision_score, recall_score

print("Overall Precision:",precision_score(y_test, predictions))
print("Overall Recall:",recall_score(y_test, predictions))
```

The overall precision is 0.72424 and the overall recall is 0.603.


True positive is for the predicted label and the actual label are both 1, false positives is for the predicted lalel is 1 but the actual label is 0, the false negatives is the predicted label is 0 but the actual label is 1 and last, the true negatives is for the predicted label and the actual label are both 0.

We can put these all in a confusion matrix.

```python
from sklearn.metrics import confusion_matrix

# Print the confusion matrix
cm = confusion_matrix(y_test, predictions)
print (cm)
```

Here we go! the confusion matrix is:

![image](https://user-images.githubusercontent.com/71245576/114949538-3a0c6e00-9e1f-11eb-98fd-ae54e2b9bb96.png)

Until now, we have considered the predictions from the model as being either 1 or 0 class labels. Actually, like the logistic regression, one of the statistical machine learning algorithms, based on probability, so that we can see the probability pairs for the label when predicting the label is true, it is P(y), whereas (1-P(y)) when the label is false.

```python
y_scores = model.predict_proba(X_test)
print(y_scores)
```




