#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel


# In[2]:


train = pd.read_csv('Training.csv').dropna(axis = 1)
test = pd.read_csv('Testing.csv').dropna(axis=1)


# In[3]:


train


# In[4]:


train = train.assign(Recurrent_infection = train.prognosis )
test = test.assign(Recurrent_infection = train.prognosis)


# In[38]:


train.loc[train['Recurrent_infection'] != 'AIDS', 'Recurrent_infection'] = 0


# In[39]:


train.loc[train['Recurrent_infection'] == 'AIDS', 'Recurrent_infection'] = 1


# In[40]:


test.loc[test['Recurrent_infection'] != 'AIDS', 'Recurrent_infection'] = 0


# In[41]:


test.loc[test['Recurrent_infection'] != 'AIDS', 'Recurrent_infection'] = 1


# In[42]:


train


# In[43]:


cols_list = train.columns.tolist()


# In[44]:


test_cols_list = test.columns.tolist()


# In[45]:


cols_list


# In[46]:


test_cols_list


# In[47]:


train.corr().T


# In[48]:


train.shape


# In[49]:


test


# In[50]:


train['Recurrent_infection'] = train['Recurrent_infection'].astype(str).astype(int)


# In[51]:


test['Recurrent_infection'] = test['Recurrent_infection'].astype('int')


# In[52]:


test.info()


# In[53]:


train.describe().T


# In[54]:


train.dtypes.value_counts()


# In[55]:


train.prognosis.value_counts()


# In[56]:


X = train.drop("prognosis", axis=1)
y = train.prognosis


# In[57]:


label_encoder_y = LabelEncoder()

# Fit and transform y
y_encoded = label_encoder_y.fit_transform(y)


print(y_encoded)


# In[58]:


X


# In[59]:


X_test = test.drop("prognosis", axis=1)
y_test = test.prognosis


# In[60]:


label_encoder_y = LabelEncoder()

# Fit and transform y
y_testencoded = label_encoder_y.fit_transform(y_test)


print(y_testencoded)


# In[61]:


classification_models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_jobs=-1, random_state=666),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier()
}


# In[62]:


result_list=[]


# In[63]:


for model_name, model in classification_models.items():
    model.fit(X, y_encoded)
    y_train_pred = model.predict(X)
    y_valid_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_encoded, y_train_pred)
    valid_accuracy = accuracy_score(y_testencoded, y_valid_pred)
    result_list.append({'Model': model_name, 'Training Accuracy': train_accuracy, 'Validation Accuracy': valid_accuracy})


# In[64]:


# Convert the list of dictionaries into a DataFrame
result_df = pd.DataFrame(result_list)

# Display the DataFrame
display(result_df)


# In[65]:


def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))


# In[66]:


for model_name in classification_models:
    model = classification_models[model_name]
    scores = cross_val_score(model, X, y_encoded, cv = 10, 
                             n_jobs = -1, 
                             scoring = cv_scoring)
    
    print("=="*30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")


# In[67]:


# Training and testing Logistic Regression
lg_model = LogisticRegression()
lg_model.fit(X, y_encoded)
preds = lg_model.predict(X_test)
 
print(f"Accuracy on train data by Logistic Regression: {accuracy_score(y_encoded, lg_model.predict(X))*100}")
 
print(f"Accuracy on test data by Logistic Regression: {accuracy_score(y_testencoded, preds)*100}")
cf_matrix = confusion_matrix(y_testencoded, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Logistic Regression on Test Data")
plt.show()
#---------------------------------------------------------------------------------------------------------------
# Training and testing Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X, y_encoded)
preds = dt_model.predict(X_test)
 
print(f"Accuracy on train data by Decision Tree: {accuracy_score(y_encoded, dt_model.predict(X))*100}")
 
print(f"Accuracy on test data by Decision Tree: {accuracy_score(y_testencoded, preds)*100}")
cf_matrix = confusion_matrix(y_testencoded, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Decision Tree on Test Data")
plt.show()

#---------------------------------------------------------------------------------------------------------------
# Training and testing Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X, y_encoded)
preds = rf_model.predict(X_test)
 
print(f"Accuracy on train data by Random Forest: {accuracy_score(y_encoded, rf_model.predict(X))*100}")
 
print(f"Accuracy on test data by Random Forest: {accuracy_score(y_testencoded, preds)*100}")
cf_matrix = confusion_matrix(y_testencoded, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest on Test Data")
plt.show()

#---------------------------------------------------------------------------------------------------------------
# Training and testing Gradient Boosting
gbc_model = GradientBoostingClassifier()
gbc_model.fit(X, y_encoded)
preds = gbc_model.predict(X_test)
 
print(f"Accuracy on train data by Gradient Boosting: {accuracy_score(y_encoded, gbc_model.predict(X))*100}")
 
print(f"Accuracy on test data by Gradient Boosting: {accuracy_score(y_testencoded, preds)*100}")
cf_matrix = confusion_matrix(y_testencoded, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Gradient Boosting on Test Data")
plt.show()

#---------------------------------------------------------------------------------------------------------------
# Training and testing SVM
sv_model = SVC()
sv_model.fit(X, y_encoded)
preds = sv_model.predict(X_test)
 
print(f"Accuracy on train data by SVM: {accuracy_score(y_encoded, sv_model.predict(X))*100}")
 
print(f"Accuracy on test data by SVM: {accuracy_score(y_testencoded, preds)*100}")
cf_matrix = confusion_matrix(y_testencoded, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM on Test Data")
plt.show()

#---------------------------------------------------------------------------------------------------------------
# Training and testing Naive Bayes
gb_model = GaussianNB()
gb_model.fit(X, y_encoded)
preds = gb_model.predict(X_test)
 
print(f"Accuracy on train data by Naive Bayes: {accuracy_score(y_encoded, gb_model.predict(X))*100}")
 
print(f"Accuracy on test data by Naive Bayes: {accuracy_score(y_testencoded, preds)*100}")
cf_matrix = confusion_matrix(y_testencoded, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Naive Bayes on Test Data")
plt.show()

#---------------------------------------------------------------------------------------------------------------
# Training and testing Neural Network
mlp_model = MLPClassifier()
mlp_model.fit(X, y_encoded)
preds = mlp_model.predict(X_test)
 
print(f"Accuracy on train data by Neural Network: {accuracy_score(y_encoded, mlp_model.predict(X))*100}")
 
print(f"Accuracy on test data by Neural Network: {accuracy_score(y_testencoded, preds)*100}")
cf_matrix = confusion_matrix(y_testencoded, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Neural Network on Test Data")
plt.show()


# In[68]:



symptoms = X.columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
   symptom = " ".join([i.capitalize() for i in value.split("_")])
   symptom_index[symptom] = index

data_dict = {
   "symptom_index":symptom_index,
   "predictions_classes":label_encoder_y.classes_
}


# In[69]:


symptom_index


# In[70]:


# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
     
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
         
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
     
    # generating individual outputs
    lg_model_prediction = data_dict["predictions_classes"][lg_model.predict(input_data)[0]]
    dt_model_prediction = data_dict["predictions_classes"][dt_model.predict(input_data)[0]]
    rf_model_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
    gbc_model_prediction = data_dict["predictions_classes"][gbc_model.predict(input_data)[0]]
    sv_model_prediction = data_dict["predictions_classes"][sv_model.predict(input_data)[0]]
    gb_model_prediction = data_dict["predictions_classes"][gb_model.predict(input_data)[0]]
    mlp_model_prediction = data_dict["predictions_classes"][mlp_model.predict(input_data)[0]]
      
    predictions = {
        "logistic_Regression_prediction": lg_model_prediction,
        "Decision Tree prediction": dt_model_prediction,
        "Random Forest": rf_model_prediction,
        "Gradient Boosting": gbc_model_prediction,
        "svm_model_prediction": sv_model_prediction,
        "Naive Bayes": gb_model_prediction,
        "Neural Network": mlp_model_prediction
    }
    
    return predictions


# In[78]:


result =  predictDisease('Muscle Wasting,Fatigue,Weight Loss,Lethargy,Loss Of Appetite,Mild Fever,Redness Of Eyes,Recurrent Infection')


# In[79]:


result

