#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.tail()


# In[6]:


print('Number of rows:',df.shape[0])
print('Number of columns:',df.shape[1])


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


# % of missing values in each column
(df.isnull().sum()/len(df)*100)


# In[10]:


# handling the missing values,we will drop the rows with missing value < 5%
df = df.drop('Loan_ID',axis=1)


# In[11]:


df.head()


# In[12]:


df.columns


# In[13]:


columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term',]


# In[14]:


df = df.dropna(subset=columns)


# In[15]:


(df.isnull().sum()/len(df)*100)


# In[16]:


df['Self_Employed'].mode()[0]


# In[17]:


df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[18]:


(df.isnull().sum()/len(df)*100)


# In[19]:


df['Credit_History'].unique()


# In[20]:


df['Self_Employed'].unique()


# In[21]:


# replacing missing value with most frequent value.
df['Credit_History'].mode()[0]


# In[22]:


# since both the columns contains categorical values , so we will use mode for missing values.
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])


# In[23]:


(df.isnull().sum()/len(df)*100)


# In[24]:


df.sample(5)


# In[25]:


df['Dependents'].unique()


# In[26]:


df['Dependents'] = df['Dependents'].replace(to_replace='3+',value='4')


# In[27]:


df['Dependents'].unique()


# In[28]:


# Handling categorical columns


# In[29]:


df['Gender'] = df['Gender'].map({'Male':1,'Female':0}).astype('int')
df['Married'] = df['Married'].map({'Yes':1,'No':0}).astype('int')
df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
df['Self_Employed'] = df['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
df['Property_Area'] = df['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0}).astype('int')


# In[30]:


df.head()


# In[31]:


df.head()


# In[32]:


# store feature matrix in X and response (tareget) in vector y


# In[33]:


X = df.drop('Loan_Status',axis=1)


# In[34]:


X


# In[35]:


y = df['Loan_Status']


# In[36]:


y


# In[37]:


# Feature Scaling


# In[38]:


cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']


# In[39]:


from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X[cols] = st.fit_transform(X[cols])


# In[40]:


X


# In[41]:


# Train-test split 
# Applying K-fold cross validation


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np


# - yaha se correction kiya h

# In[43]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y  ,test_size = 0.33 , random_state = 42)


# In[44]:


X_train.shape , y_train.shape , X_test.shape , y_test.shape 


# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import recall_score , classification_report , confusion_matrix  ,roc_curve , roc_auc_score , accuracy_score
from sklearn.metrics import precision_recall_curve , auc ,f1_score , plot_confusion_matrix , precision_score , recall_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# In[46]:


model_list = []
accuracy_list = []
recall_list = []
precision_list = []
f1_score_list= [] 


# In[47]:


def Model_features(X_train , y_train , X_test , y_test , y_pred , classifier  , model_name):
#     fig ,ax = plt.subplots(figsize = (7,6))
    accuracy , precision , recall , f1_s = round(accuracy_score(y_test , y_pred) , 3) , round(precision_score(y_test, y_pred, average="micro")  ,3), round(recall_score(y_test , y_pred) ,3), round(f1_score(y_test , y_pred) , 3)
    print(f'Accuracy Score is :{accuracy}')
    print(f'Precision Score is :{precision}')
    print(f'Recall Score is :{recall}')
    print(f'f1  Score is :{f1_s}')
    model_list.append(model_name)
    accuracy_list.append(accuracy)
    recall_list.append(recall)
    precision_list.append(precision)
    f1_score_list.append(f1_s)
    
#     print(f'f1  Score is :{round(specificity_score(y_test , y_pred) , 3)}')
    print(metrics.classification_report(y_test, y_pred)) 


# model_df = {}
# def model_val(model,X,y):
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
#     model.fit(X_train,y_train)
#     y_pred = model.predict(X_test)
#     print(f"{model} accuarcy is {accuracy_score(y_test,y_pred)}")
#     
#     score = cross_val_score(model,X,y,cv=10)
#     print(f"{model} average cross val score is {np.mean(score)}")
#     model_df[model] = round(np.mean(score)*100,2)

# - ** Feature Importance

# In[48]:


# Define a function that plots the feature weights for a classifier.
def feature_weights(X_df, classifier, classifier_name):
    weights = round(pd.Series(classifier.coef_[0], index=X_df.columns.values).sort_values(ascending=False) ,2 )
    
    top_weights_selected = weights[:5]
    plt.figure(figsize=(7,6))
    plt.tick_params(labelsize=10)#plt.xlabel(fontsize=10)
    plt.title(f'{classifier_name} - Top 5 Features')
    ax = top_weights_selected.plot(kind="bar")
    ax.bar_label(ax.containers[0])
    
    return print("")


# In[49]:


def confusion_matrix_plot(X_test , y_test , classifier ,classifier_name):
    ax = plot_confusion_matrix(classifier, X_test, y_test, display_labels=["No Personal Loan", "Personal Loan"], cmap=plt.cm.Blues, normalize='true')


# 1). Logistic Regression
# -

# In[50]:


model_lr= LogisticRegression(random_state=0)  
model_lr.fit(X_train, y_train) 
y_pred = model_lr.predict(X_test)
model_lr.score(X_test , y_test)


# In[51]:


Model_features(X_train , y_train , X_test , y_test  , y_pred , model_lr , "Logistic Reegression")   
feature_weights(X_train , model_lr , "Logistic Regression")
confusion_matrix_plot(X_test , y_test , model_lr , "Logistic Regression")


# 2). Support Vector Machine
# -

# In[52]:


svm = SVC(kernel='rbf', probability=True) 
svm.fit(X_train,y_train)

# Make predictions (classes and probabilities) with the trained model on the test set.
y_pred = svm.predict(X_test)
svm.score(X_test , y_test)  


# In[53]:


Model_features(X_train , y_train , X_test , y_test  , y_pred , svm , "Support Vector Machine")   
confusion_matrix_plot(X_test , y_test , svm , "Support Vector Machine")


# 3). Random Forest Classifier
# -

# In[54]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf.score(X_test , y_test) 


# In[55]:


Model_features(X_train , y_train , X_test , y_test  , y_pred , rf , "Random Forest Classifier")   
confusion_matrix_plot(X_test , y_test , rf , "Random Forest Classifier")


# Hyperparameter Tuning
# -

# 1). For Logistic regression
# -

# In[56]:


from sklearn.model_selection import RandomizedSearchCV


# In[57]:


log_reg_grid = {"C":np.logspace(-4,4,20),
               "solver":['liblinear']}


# In[58]:


rs_log_reg = RandomizedSearchCV(LogisticRegression(),param_distributions=log_reg_grid,
                               n_iter=20,cv=5,verbose=True)


# In[59]:


rs_log_reg.fit(X,y)


# In[60]:


rs_log_reg.best_score_


# In[61]:


rs_log_reg.best_params_


# 2). For SVC
# -

# In[62]:


from sklearn import svm
model = svm.SVC()


# In[63]:


svc_grid = {'C':[0.25,0.50,0.75,1],'kernel':['linear']}


# In[64]:


rs_svc = RandomizedSearchCV(svm.SVC(),
                  param_distributions=svc_grid,cv=5,n_iter=20,verbose=True
                  )


# In[65]:


rs_svc.fit(X,y)


# In[66]:


rs_svc.best_score_


# In[67]:


rs_svc.best_params_


# 3). For Random Forest Classifier
# -

# In[68]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[69]:


RandomForestClassifier()


# In[70]:


rf_grid = {'n_estimators':np.arange(10,1000,10),
          'max_features':['auto','sqrt'],
           'max_depth':[None,3,5,10,20,30],
          'min_samples_split':[2,5,20,50,100],
          'min_samples_leaf':[1,2,5,10]}


# In[71]:


rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                  param_distributions=rf_grid,cv=5,n_iter=20,verbose=True
                  )


# In[72]:


rs_rf.fit(X,y, groups=None)


# In[73]:


rs_rf.best_score_


# In[74]:


rs_rf.best_params_


# - Logistic Regression score before Hyperparameter tuning : 80.48
# - Logistic Regression score after Hyperparameter tuning : 80.48

# - SVC score before Hyperparameter tuning : 79.38
# - SVC score after Hyperparameter tuning : 80.66

# - RandomForestClassifier score before Hyperparameter tuning : 77.76
# - RandomForestClassifier score after Hyperparameter tuning : 80.66

# - The above result shows that the performance of model has been increased after using Hyperparameter Tuning.

# - Let's use Random Forest Classifier for our production.

# Save The Model
# -

# In[75]:


X = df.drop('Loan_Status',axis=1)
y = df['Loan_Status']


# In[76]:


rf = RandomForestClassifier(n_estimators = 210,
 min_samples_split = 20,
 min_samples_leaf = 5,
 max_features = 'sqrt',
 max_depth = 5) 


# In[77]:


rf.fit(X,y)


# In[78]:


import joblib


# In[79]:


joblib.dump(rf,'loan_status_predict')


# In[80]:


model = joblib.load('loan_status_predict')


# In[81]:


import pandas as pd
df_new = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2889,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1
},index=[0])


# In[82]:


df_new


# In[83]:


result = model.predict(df_new)


# In[84]:


if result==1:
    print('Loan Approved')
else:
    print('Loan Not Approved')


# GUI
# -

# In[85]:


from tkinter import *
import joblib
import pandas as pd


# In[ ]:


def show_entry():
    
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())
    p8 = float(e8.get())
    p9 = float(e9.get())
    p10 = float(e10.get())
    p11 = float(e11.get())
    
    model = joblib.load('loan_status_predict')
    df = pd.DataFrame({
    'Gender':p1,
    'Married':p2,
    'Dependents':p3,
    'Education':p4,
    'Self_Employed':p5,
    'ApplicantIncome':p6,
    'CoapplicantIncome':p7,
    'LoanAmount':p8,
    'Loan_Amount_Term':p9,
    'Credit_History':p10,
    'Property_Area':p11
},index=[0])
    result = model.predict(df)
    
    if result == 1:
        Label(master, text="Loan approved").grid(row=31)
    else:
        Label(master, text="Loan Not Approved").grid(row=31)
        
    
master =Tk()
master.title("Loan Status Prediction Using Machine Learning")
label = Label(master,text = "Loan Status Prediction",bg = "black",
               fg = "white").grid(row=0,columnspan=2)

Label(master,text = "Gender [1:Male ,0:Female]").grid(row=1)
Label(master,text = "Married [1:Yes,0:No]").grid(row=2)
Label(master,text = "Dependents [1,2,3,4]").grid(row=3)
Label(master,text = "Education").grid(row=4)
Label(master,text = "Self_Employed").grid(row=5)
Label(master,text = "ApplicantIncome").grid(row=6)
Label(master,text = "CoapplicantIncome").grid(row=7)
Label(master,text = "LoanAmount").grid(row=8)
Label(master,text = "Loan_Amount_Term").grid(row=9)
Label(master,text = "Credit_History").grid(row=10)
Label(master,text = "Property_Area").grid(row=11)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)


e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)
e7.grid(row=7,column=1)
e8.grid(row=8,column=1)
e9.grid(row=9,column=1)
e10.grid(row=10,column=1)
e11.grid(row=11,column=1)

Button(master,text="Predict",command=show_entry).grid()

mainloop()

