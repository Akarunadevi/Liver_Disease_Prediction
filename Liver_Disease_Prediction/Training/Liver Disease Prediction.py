import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv('D:/A_Review_of_Liver_Patient_Analysis_Methods_Using _Machine_Learning/Dataset/indian_liver_patient.csv')
print(data.head())
print(data.info())
print(data.isnull().any())
print(data.isnull().sum())
data.fillna(data.mode().iloc[0], inplace=True)
print(data.isnull().sum())
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data['Gender']=lc.fit_transform(data['Gender'])
sns.histplot(data['Age'])

plt.title('Age Distribution Graph')
plt.show()
sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap')
plt.show()
from sklearn.preprocessing import scale
x_scaled = pd.DataFrame(scale(data), columns=data.columns)
print(x_scaled.head())
# separate the features and target variable
x = data.iloc[:,:-1]
y = data.Dataset

# standardize the features using StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Import required libraries
from imblearn.over_sampling import SMOTE

# Create a SMOTE object
smote = SMOTE()

# Check the class distribution in the training data
print("Before SMOTE:")
print(y_train.value_counts())

# Use SMOTE to oversample the minority class in the training data
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# Check the class distribution after oversampling
print("After SMOTE:")
print(y_train_smote.value_counts())



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Assuming you have already split your data into x_train_smote, y_train_smote, x_test, and y_test

# Initialize the model
model1 = RandomForestClassifier()

# Fit the model on the training data
model1.fit(x_train_smote, y_train_smote)

# Use the model to predict on the test data
y_pred = model1.predict(x_test)

# Calculate the accuracy of the model
rfc1= accuracy_score(y_test, y_pred)
print("Accuracy:", rfc1)

# Print a confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_matrix)

# Print a classification report
classification_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report)


# Import required libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Create a decision tree classifier object
model4 = DecisionTreeClassifier()

# Train the model on the training data
model4.fit(x_train_smote, y_train_smote)

# Use the trained model to predict on the test data
y_predict = model4.predict(x_test)

# Evaluate the accuracy of the model
dtc1= accuracy_score(y_test, y_predict)
print("Decision Tree Model Accuracy: ", dtc1)

# Print the confusion matrix
print(pd.crosstab(y_test, y_predict))

# Print the classification report
print(classification_report(y_test, y_predict))


# Import required libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Create a KNN classifier object
model2 = KNeighborsClassifier()

# Train the model on the training data
model2.fit(x_train_smote, y_train_smote)

# Use the trained model to predict on the test data
y_predict = model2.predict(x_test)

# Evaluate the accuracy of the model
knn1= accuracy_score(y_test, y_predict)
print("KNN Model Accuracy: ", knn1)

# Print the confusion matrix
print(pd.crosstab(y_test, y_predict))

# Print the classification report
print(classification_report(y_test, y_predict))


# Import required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Create a logistic regression classifier object
model5 = LogisticRegression()

# Train the model on the training data
model5.fit(x_train_smote, y_train_smote)

# Use the trained model to predict on the test data
y_predict = model5.predict(x_test)

# Evaluate the accuracy of the model
logi1= accuracy_score(y_test, y_predict)
print("Logistic Regression Model Accuracy: ", logi1)

# Print the confusion matrix
print(pd.crosstab(y_test, y_predict))

# Print the classification report
print(classification_report(y_test, y_predict))

# Import required libraries
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a Sequential model
classifier = Sequential()

# Add layers to the model
classifier.add(Dense(units=100, activation='relu', input_dim=10))
classifier.add(Dense(units=50, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model_history = classifier.fit(x_train, y_train, batch_size=100, validation_split=0.2, epochs=100)


def predict_exit(sample_value):
    sample_value = np.array(sample_value)
    sample_value = sample_value.reshape(1,-1)
    sample_value = scale(sample_value)
    return classifier.predict(sample_value)

sample_value = [[50,1,1.2,0.8,150,70,80,7.2,3.4,0.8]]

if predict_exit(sample_value) > 0.5:
    print('Prediction: Liver Patient')
else:
    print('Prediction: Healthy')


acc_smote = [['KNN Classifier', knn1], ['RandomForestClassifier', rfc1],
             ['DecisionTreeClassifier', dtc1], ['LogisticRegression', logi1]]

Liverpatient_pred = pd.DataFrame(acc_smote, columns=['Classification Models', 'Accuracy Score'])
print(Liverpatient_pred)

import matplotlib.pyplot as plt
import seaborn as sns

# set plot size
plt.figure(figsize=(7, 5))

# rotate x-axis labels by 90 degrees
plt.xticks(rotation=90)

# set plot title
plt.title('Classification Models & Accuracy Scores after SMOTE', fontsize=18)

# create bar plot
sns.barplot(x='Classification Models', y='Accuracy Score', data=Liverpatient_pred, palette='Set2')

# show plot
plt.show()


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import pandas as pd

model = ExtraTreesClassifier()
model.fit(x, y)

feature_importances = pd.DataFrame(model.feature_importances_, index=x.columns, columns=['Importance']).sort_values('Importance', ascending=False)
print(feature_importances)

df = pd.DataFrame(model.feature_importances_, index=x.columns).sort_values(0, ascending=False)

plt.figure(figsize=(7,6))
plt.barh(df.index, df[0], align='center')
plt.title("FEATURE IMPORTANCE", fontsize=14)
plt.show()


import joblib
joblib.dump(model1, 'ETC.pkl')
joblib.dump(model, 'ETC.joblib')

