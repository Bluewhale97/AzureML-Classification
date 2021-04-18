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
The decision to score a prediction as a 1 or a 0 depends on the thredshold to which the predicted probabilities are compared, see the result:

![image](https://user-images.githubusercontent.com/71245576/115152437-0f006500-a03f-11eb-8025-69a15c1f5c17.png)

Here is a common way to evaluate a classifier that examine the true positive rate and the false positive rate for a range of possible thresholds. These rates then are plotted against all possible thresholds to form a chart known as a received operator characteristic (ROG) chart:

```python
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

The ROC chart shows the curve of the true and false positive rates for different threshold values between 0 and 1. A perfect classifier would have a curve that goes straight up the left side and straight across the top.

![image](https://user-images.githubusercontent.com/71245576/115152603-c006ff80-a03f-11eb-9fe7-18c6561138db.png)

The area under the curve(AUC) is a value between 0 and 1 that quantifies the overall performance of the model. The closer to 1 this value is the better the model.

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
```
The AUC is 0.8568.

## 2. Preprocessing for better performance

In the above case the ROC curve and the AUC indicate that the model perfroms better than a random guess. In practive, we should perform preprocessing of the data to make it easier for the algorithm to fit a model. We talk a little in the previous discussion in regression part, the scaling and factorization are very common in preprocessing process. 

There is a feature in Scikit-Learn, called pipelines. These enable us to define a set of preprocessing steps that end up with an algorithm. Personally, it helps us build up the pipelines of preprocessing and combine them in creating preprocessing and training pipeline, consequently, fit a model using pipeline function.

Let's see how it represents:

```python
# Train the model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define preprocessing for numeric columns (normalize them so they're on the same scale)
numeric_features = [0,1,2,3,4,5,6]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Define preprocessing for categorical features (encode the Age column)
categorical_features = [7]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('logregressor', LogisticRegression(C=1/reg, solver="liblinear"))])


# fit the pipeline to train a logistic regression model on the training set
model = pipeline.fit(X_train, (y_train))
print (model)
```

The pipeline encapsulates the preprocessing steps as well as model training. Now get predictions from test data:

```python
# Get predictions from test data
predictions = model.predict(X_test)
y_scores = model.predict_proba(X_test)
```
Evaluation metrics:
```python
# Get evaluation metrics
cm = confusion_matrix(y_test, predictions)
print ('Confusion Matrix:\n',cm, '\n')
print('Accuracy:', accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions))
print("Overall Recall:",recall_score(y_test, predictions))
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))

# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```
It shows the confision metrics, accuracy, overall precision and recall, ROC/AUC:

![image](https://user-images.githubusercontent.com/71245576/115153065-f776ab80-a041-11eb-80a1-51674602e894.png)

The results look a little better, so preprocessing the data makde a difference!

Now let's try a different algorithm, we would perform a random forest:

```python
from sklearn.ensemble import RandomForestClassifier

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('logregressor', RandomForestClassifier(n_estimators=100))])

# fit the pipeline to train a random forest model on the training set
model = pipeline.fit(X_train, (y_train))
print (model)

predictions = model.predict(X_test)
y_scores = model.predict_proba(X_test)
cm = confusion_matrix(y_test, predictions)
print ('Confusion Matrix:\n',cm, '\n')
print('Accuracy:', accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions))
print("Overall Recall:",recall_score(y_test, predictions))
auc = roc_auc_score(y_test,y_scores[:,1])
print('\nAUC: ' + str(auc))
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```
The random forest makes better than logistic regression. 

![image](https://user-images.githubusercontent.com/71245576/115153250-ed08e180-a042-11eb-9442-bfb2d884b0b7.png)

When we think a model is reasonably useful. we can save it for the latter use to predict labels for new data.

```python
import joblib

# Save the model as a pickle file
filename = './models/diabetes_model.pkl'
joblib.dump(model, filename)
```
When we have some new observations for which the label is unknown, we can load the model and use it to predict values for unknown label:

```python
# Load the model from the file
model = joblib.load(filename)

# predict on a new sample
# The model accepts an array of feature arrays (so you can predict the classes of multiple patients in a single call)
# We'll create an array with a single array of features, representing one patient
X_new = np.array([[2,180,74,24,21,23.9091702,1.488172308,22]])
print ('New sample: {}'.format(list(X_new[0])))

# Get a prediction
pred = model.predict(X_new)

# The model returns an array of predictions - one for each set of features submitted
# In our case, we only submitted one patient, so our prediction is the first one in the resulting array.
print('Predicted class is {}'.format(pred[0]))
```
See the predicted values:

![image](https://user-images.githubusercontent.com/71245576/115153416-bed7d180-a043-11eb-9a96-70cc2c315ceb.png)

## 3. Multiclass classification

Binary classification is made for preidcting the label of two classes or categories, when there are more than two classes in the feature, we could use multiclass classification, what should be noticed is that the multiclass classification can be considered of as a combination of multiple binary classifiers. There are two ways to consider:

a. One vs Rest(OVR): when predicting a class is true, others are all false.
b. One vs One: when predict a class is true, a specified another one is as false.

Let's try a practice.

First we need to explore the data, the data we will use is the data set that contains observations of three different species of penguin, which is a subset of data collected and made availale by Dr.Kristen Gorman and the Palmer Station, Antarctica LTER.

```python
import pandas as pd

# load the training dataset
penguins = pd.read_csv('data/penguins.csv')

# Display a random sample of 10 observations
sample = penguins.sample(10)
sample
```

See the 10 of a sample, Sepcies column is absolutely the label, 0, 1, and 2 is the possible species that includes.

![image](https://user-images.githubusercontent.com/71245576/115153637-1591db00-a045-11eb-8cfb-3129a1b41d8f.png)

Curiously, the actual species names are reveraled:
```python
penguin_classes = ['Adelie', 'Gentoo', 'Chinstrap']
print(sample.columns[0:5].values, 'SpeciesName')
for index, row in penguins.sample(10).iterrows():
    print('[',row[0], row[1], row[2], row[3], int(row[4]),']',penguin_classes[int(row[4])])
```
Now we knew the names of the species: Chinstrap, Adelie, and Gentoo.

Let's see if there are any missing values:
```python
# Count the number of null values for each column
penguins.isnull().sum()
```
It looks like there are some missing feature values but no missing labels:

![image](https://user-images.githubusercontent.com/71245576/115153818-dca63600-a045-11eb-81e3-62717e5be615.png)

Dig a little deeper and see the rows that contain nulls:

```python
# Show rows containing nulls
penguins[penguins.isnull().any(axis=1)]
```
Two rows that contain no feature values at all, we would discard them from the data set.

![image](https://user-images.githubusercontent.com/71245576/115153864-01021280-a046-11eb-8f44-cbabc227bbf4.png)

Discard them and confirm that there are now no nulls.

```python
# Drop rows containing NaN values
penguins=penguins.dropna()
#Confirm there are now no nulls
penguins.isnull().sum()
```

Now let's create box plots to know the correlation of the features.

```python
from matplotlib import pyplot as plt
%matplotlib inline

penguin_features = ['CulmenLength','CulmenDepth','FlipperLength','BodyMass']
penguin_label = 'Species'
for col in penguin_features:
    penguins.boxplot(column=col, by=penguin_label, figsize=(6,6))
    plt.title(col)
plt.show()
```
From the box plots, it looks like species 0 and 2 (Amelie and Chinstrap) have similar data profiles for culmen depth, flipper length, and body mass, but Chinstraps tend to have longer culmens. Species 1 (Gentoo) tends to have fairly clearly differentiated features from the others; which should help us train a good classification model.

Now separate features and label and then split the data into subsets for training and validation:
```python
from sklearn.model_selection import train_test_split

# Separate features and labels
penguins_X, penguins_y = penguins[penguin_features].values, penguins[penguin_label].values

# Split data 70%-30% into training set and test set
x_penguin_train, x_penguin_test, y_penguin_train, y_penguin_test = train_test_split(penguins_X, penguins_y,
                                                                                    test_size=0.30,
                                                                                    random_state=0,
                                                                                    stratify=penguins_y)

print ('Training Set: %d, Test Set: %d \n' % (x_penguin_train.shape[0], x_penguin_test.shape[0]))
```

The number of observations in training set 239, 103 in test set. Try to train and evaluate a multiclass classifier:

```python
from sklearn.linear_model import LogisticRegression

# Set regularization rate
reg = 0.1

# train a logistic regression model on the training set
multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000).fit(x_penguin_train, y_penguin_train)
print (multi_model)
```

Use the trained model to predict the labels for the test features, and compare the predicted labels to the actual labels:

```python
penguin_predictions = multi_model.predict(x_penguin_test)
print('Predicted labels: ', penguin_predictions[:15])
print('Actual labels   : ' ,y_penguin_test[:15])
```

See the part of prediction:

![image](https://user-images.githubusercontent.com/71245576/115154111-2e02f500-a047-11eb-863e-104f1c93ade9.png)

Let's look at a classification report:

```python
from sklearn. metrics import classification_report

print(classification_report(y_penguin_test, penguin_predictions))
```

As with binary classification, the report includes precision and recall metrics for each class. However, while with binary classification we could focus on the scores for the positive class; in this case, there are multiple classes so we need to look at an overall metric (either the macro or weighted average) to get a sense of how well the model performs across all three classes.

![image](https://user-images.githubusercontent.com/71245576/115154152-6dc9dc80-a047-11eb-90db-9edf9a64f8b9.png)

We also can get the overall metrics separately from the report using the scikit-lean metrics score classes:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Overall Accuracy:",accuracy_score(y_penguin_test, penguin_predictions))
print("Overall Precision:",precision_score(y_penguin_test, penguin_predictions, average='macro'))
print("Overall Recall:",recall_score(y_penguin_test, penguin_predictions, average='macro'))
```

![image](https://user-images.githubusercontent.com/71245576/115154184-918d2280-a047-11eb-9a39-1bd0df1f2db1.png)

The overall accuracy is 0.9708, precision is 0.9688 and the recall is 0.9608. We now look at the confusion matrix for our model:

```python
from sklearn.metrics import confusion_matrix

# Print the confusion matrix
mcm = confusion_matrix(y_penguin_test, penguin_predictions)
print(mcm)
```

The confusion matrix shows the intersection of predicted and actual label values for each class:

![image](https://user-images.githubusercontent.com/71245576/115154350-62c37c00-a048-11eb-83c1-4eccb7bdf3bb.png)

What we should now is that when dealing with multiple classes, it is generally more intuitive to visualize this as a heat map, like this:

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(penguin_classes))
plt.xticks(tick_marks, penguin_classes, rotation=45)
plt.yticks(tick_marks, penguin_classes)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()
```

The darker the color, the more the numers of cases.

![image](https://user-images.githubusercontent.com/71245576/115154405-af0ebc00-a048-11eb-824a-6c6258fb0a00.png)

In the case of a multiclass classification model, a single ROC curve showing true positive rate vs false positive rate is not possible. However, you can use the rates for each class in a One vs Rest (OVR) comparison to create a ROC chart for each class.

```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Get class probability scores
penguin_prob = multi_model.predict_proba(x_penguin_test)

# Get ROC metrics for each class
fpr = {}
tpr = {}
thresh ={}
for i in range(len(penguin_classes)):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_penguin_test, penguin_prob[:,i], pos_label=i)
    
# Plot the ROC chart
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label=penguin_classes[0] + ' vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label=penguin_classes[1] + ' vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label=penguin_classes[2] + ' vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
```

The result shows:

![image](https://user-images.githubusercontent.com/71245576/115154474-07de5480-a049-11eb-8eb0-8f862e3bd950.png)

To quantify the ROC performance, we can calculate an aggregate area under the curve score that is averaged across all of the OVR curves.

```python
auc = roc_auc_score(y_penguin_test,penguin_prob, multi_class='ovr')
print('Average AUC:', auc)
```

The average AUC is 0.998. 

Actually, after evaluating metrics of performance, the model performance is good. Like with binary classification, we also can use a pipeline to apply preprocessing steps to the data before fitting it to an algorithm to train a model.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Define preprocessing for numeric columns (scale them)
feature_columns = [0,1,2,3]
feature_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
    ])

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('preprocess', feature_transformer, feature_columns)])

# Create training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', SVC(probability=True))])


# fit the pipeline to train a linear regression model on the training set
multi_model = pipeline.fit(x_penguin_train, y_penguin_train)
print (multi_model)
```

Now, evaluate the new model:
```python
# Get predictions from test data
penguin_predictions = multi_model.predict(x_penguin_test)
penguin_prob = multi_model.predict_proba(x_penguin_test)

# Overall metrics
print("Overall Accuracy:",accuracy_score(y_penguin_test, penguin_predictions))
print("Overall Precision:",precision_score(y_penguin_test, penguin_predictions, average='macro'))
print("Overall Recall:",recall_score(y_penguin_test, penguin_predictions, average='macro'))
print('Average AUC:', roc_auc_score(y_penguin_test,penguin_prob, multi_class='ovr'))

# Confusion matrix
plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(penguin_classes))
plt.xticks(tick_marks, penguin_classes, rotation=45)
plt.yticks(tick_marks, penguin_classes)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()
```

The result shows:

![image](https://user-images.githubusercontent.com/71245576/115154581-9bb02080-a049-11eb-85dc-68604390bcb1.png)

Save the model for later use.
```python
import joblib

# Save the model as a pickle file
filename = './models/penguin_model.pkl'
joblib.dump(multi_model, filename)
```
When we get a new set of penguin observation, we can use it to predict the class of a new penguin observation:

```python
# Load the model from the file
multi_model = joblib.load(filename)

# The model accepts an array of feature arrays (so you can predict the classes of multiple penguin observations in a single call)
# We'll create an array with a single array of features, representing one penguin
x_new = np.array([[50.4,15.3,224,5550]])
print ('New sample: {}'.format(x_new[0]))

# The model returns an array of predictions - one for each set of features submitted
# In our case, we only submitted one penguin, so our prediction is the first one in the resulting array.
penguin_pred = multi_model.predict(x_new)[0]
print('Predicted class is', penguin_classes[penguin_pred])
```

The result:

![image](https://user-images.githubusercontent.com/71245576/115154653-03666b80-a04a-11eb-8477-e6df3a929802.png)

You also can submit a batch of penguin observations to the model:

```python
# This time our input is an array of two feature arrays
x_new = np.array([[49.5,18.4,195, 3600],
         [38.2,20.1,190,3900]])
print ('New samples:\n{}'.format(x_new))

# Call the web service, passing the input data
predictions = multi_model.predict(x_new)

# Get the predicted classes.
for prediction in predictions:
    print(prediction, '(' + penguin_classes[prediction] +')')
```

The result:

![image](https://user-images.githubusercontent.com/71245576/115154690-2a24a200-a04a-11eb-93e6-bca3e25ac423.png)

## Reference

Train and evaluate classification models, retrieved from https://docs.microsoft.com/en-us/learn/modules/train-evaluate-classification-models/



