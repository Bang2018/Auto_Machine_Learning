# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:20:07 2020
This tutorial gives you thorough understanding of the followings:
    1. Auto EDA
    2. Auto ML
    3. Graphviz and external commands in Python
    4. Frequency Table,Two-Way Table,Two-Way Table -Joint Probability,Two-Way Table -Marginal Probability,and Two-Way Table -Conditional Probability for Categorical Variable
    5. Cross-Validation and it's use
    6. Bias and Variance
    7. Learning Curve
    8. Use of Logistic Regression, Decision Tree, Naive Bayes, Stochastic Gradient,Random Forest,K Nearest Neighbor,Linear SVC,and Perceptron for Binary Classification
    
Outcome:
    1. Automatic Selection of Numeric and Categorical Variable
    2. Learning Curve Plot for Different Classifiers
    3. Decision Tree Plot
    4. Automatic Selection of Best Classifier
    
System Requirment:
    1. I have used Win 32 bit System
    2. Python Version 3.6
    3. Install GraphViz
    
Dataset: Titanic Dataset - Download from Kaggle

Acknowledgement: "Predicting the Survival of Titanic Passengers" by Niklas Donges
 
    
@author: Krishnendu Mukherjee
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys

folder_path = "D:/Consultancy/DS Train/Lec16 DecisionTree Classification/Data/"
file = ["train.csv","test.csv"]

def pandas_intro(tdf,dpath,file):
    ''' This function will plot horizontal bar
        and show the file size, and categorical
        and numeric attributes'''
    n_col = tdf.shape[1]
    n_cat_var=0
    n_num_var=0
    for col in tdf.columns:
        if(tdf.dtypes[col]==np.object):
            n_cat_var=n_cat_var+1
        if(tdf.dtypes[col]==np.int64 or tdf.dtypes[col]==np.float64):
            n_num_var=n_num_var+1
    n_num_var=round(n_num_var*100/n_col,2)
    n_cat_var=round(n_cat_var*100/n_col,2)
    var_lst=[n_cat_var,n_num_var]
    var_name=["Categorical","Numeric"]
    df = pd.DataFrame({"Variables":var_name,"% in Dataset":var_lst})
    df_sort = df.sort_values(by="% in Dataset",ascending=False).set_index("Variables")
    plt.figure()
    df_sort.plot.barh(color="grey",alpha=0.8,figsize=(10,2))
    #fsize=os.stat(os.path.join(dpath,file)) - Method I to get file size
    #sizestr=str(np.ceil(fsize.st_size/1024))+" KB"
    sizestr = str(np.ceil(os.path.getsize(os.path.join(dpath,file))/1024))+" KB"
    title = "File:"+str(file)+"|"+sizestr+"|Rows:"+str(tdf.shape[0])+"|Col:"+str(tdf.shape[1])
    plt.title(title)
    
    
   

def load_data(folder_path,file,rows):
    if not os.path.isfile(os.path.join(folder_path,file[0])):
       print("No Such File/Directory..")
       flg = True
       if flg:
           fl = glob.glob(os.path.join(folder_path,"t*.csv"))
           path1 = os.path.join(folder_path,fl[0])
           path2 = os.path.join(folder_path,fl[0])
    else:
        path1 = os.path.join(folder_path,file[0])
        path2 = os.path.join(folder_path,file[1])
       
    train_df = pd.read_csv(path1)
    test_df = pd.read_csv(path2)
    pandas_intro(train_df,folder_path,file[0])
    
    print(train_df.head(rows))
    print(train_df.describe())
    #print("Dataset has",train_df.shape[0]," rows and ",train_df.shape[1]," columns")
    
    return train_df,test_df


def plot_graph(dtmodel,dforg):
    from sklearn.tree import export_graphviz
    plt.figure(figsize=(20,20))
    export_graphviz(dtmodel,out_file="dtree.dot",feature_names=dforg.columns)
    #from subprocess import call
    #call(["dot", "-Tpng", "dtree.dot", "-o","dtree.png","-Gdpi=600"])
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz2.38/bin'
    os.system('dot -Tpng dtree.dot -o dtree1.png')
    import matplotlib.image as img
    plt.imshow(img.imread("dtree1.png"))
    plt.show()
        
def automatic_EDA(df,df_test,target,dindex,droplst):
    cat_var=[]
    num_var=[]
    for col in df.columns:
        if (df.dtypes[col]==np.object):
            print("Categorical Vairable :",col)
            if col not in droplst:
               cat_var.append(col)
        if (df.dtypes[col]==np.int64 or df.dtypes[col]==np.float64):
            if col not in [target,dindex]:
               print("Numeric Variable :",col)
               num_var.append(col)
               if(df.dtypes[col]==np.int64):
                  plt.figure()
                  plt.hist(df[col])
                  plt.show()
    
    for col in num_var:  
        print("Missing Values in Numeric Variable :",col," is ",df[col].isnull().sum())
    
    for col in cat_var:  
        print("Missing Values in Categorical Vairable :",col," is ",df[col].isnull().sum())
    
    df=df.dropna()
    
             
    print("Total number of Categorical Variables:",len(cat_var))  
    print("Total number of Numberic Variables:",len(num_var))
    print("Selecting List of Categorical Variables:",cat_var)
    print("Selecting List of Numeric Variables:",num_var)
    
    print("\nFrequency Table for Categorical Variables")
    for col in cat_var:
        print("\nCategorical Variable:",col)
        print(pd.crosstab(index=df[col],columns="Count",dropna=True))
        
    print("\nTwo-way Table for Categorical Variables")
    for col in cat_var:
        print("\nCategorical Variable:",col)
        print(pd.crosstab(index=df[target],columns=df[col],dropna=True))
        
    print("\nTwo-way Table - Joint Probability for Categorical Variables")
    for col in cat_var:
        print("\nCategorical Variable:",col)
        print(pd.crosstab(index=df[target],columns=df[col],normalize=True,dropna=True))
        
    print("\nTwo-way Table - Marginal Probability for Categorical Variables")
    for col in cat_var:
        print("\nCategorical Variable:",col)
        print(pd.crosstab(index=df[target],columns=df[col],normalize=True,margins=True,dropna=True))
        
    print("\nTwo-way Table - Conditional Probability for Categorical Variables")
    for col in cat_var:
        print("\nCategorical Variable:",col)
        print(pd.crosstab(index=df[target],columns=df[col],normalize="index",margins=True,dropna=True))
        
    print("\nCorrelation for Numerical Variables")
    df_num_var = df[num_var]
    corr_matrix = df_num_var.corr()
    print(corr_matrix)
    
    for col in cat_var:
        dummy = pd.get_dummies(df[col],prefix=col)
        df = df.join(dummy)
    col_lst = [col for col in df.columns if col not in cat_var if col not in droplst]
    df=df[col_lst]
    
        
    print("\nFormatted Train Dataset..")
    print(df.head(5))
    
    cat_var_test=[]
    num_var_test =[]
    
    for col in df_test.columns:
        if (df_test.dtypes[col]== np.object):
            cat_var_test.append(col)
        if (df_test.dtypes[col] == np.int64 or df_test.dtypes[col] == np.float64):
            num_var_test.append(col)
            
    df_test = df_test.drop(dindex,axis=1)
    
    for col in cat_var_test:
        if col not in droplst:
           dummy_new = pd.get_dummies(df_test[col],prefix=col)
           df_test = df_test.join(dummy_new)
    col_test = [col for col in df_test.columns if col not in cat_var_test]    
    
    df_test = df_test[col_test]
    print("Columns:",df_test.columns)
    df_test = df_test.dropna()
       
    print("Formatted Test Dataset...")
    print(df_test.head(5))
    
    X_train = df.drop(target,axis=1)
    Y_train = df[target]
    return X_train,Y_train,df_test

def learning_curve_fit(estimator_name,estimator,Xtrain,Ytrain,cv):
    from sklearn.learning_curve import learning_curve
    
    tr_size, tr_score,val_score = learning_curve(estimator,Xtrain,Ytrain,cv=cv,train_sizes=np.linspace(0.01,1.0,50))
    tr_score_mean = np.mean(tr_score,axis=1)
    tr_score_std = np.std(tr_score,axis=1)
    val_score_mean = np.mean(val_score,axis=1)
    val_score_std = np.std(val_score,axis=1)
    plt.figure()
    plt.plot(tr_size,tr_score_mean,"--",color="red",label="Training Score")
    plt.fill_between(tr_size,tr_score_mean-tr_score_std,tr_score_mean + tr_score_std, alpha=0.1,color="red")
    plt.plot(tr_size,val_score_mean,color="green",label="Cross-Validation Score")
    plt.fill_between(tr_size,val_score_mean-val_score_std,val_score_mean + val_score_std, alpha=0.1,color="green")
    title = "Learning Curve:" + str(estimator_name)
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.show()

def Auto_ML(Xtrain,Ytrain,Xtest):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC,LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.model_selection import cross_val_score,cross_val_predict
    
    
    print("\nSelecting Logistic Regression, Decision Tree, Naive Bayes, Stochastic Gradient,Random Forest,K Nearest Neighbor,Linear SVC,and Perceptron\n")
    
    lgreg = LogisticRegression(max_iter=2000,solver="sag")
    lr = lgreg.fit(Xtrain,Ytrain)
    lgreg_acc = round(lr.score(Xtrain,Ytrain)*100,2)
    print("\nAccuracy of Logistic Regression:",lgreg_acc)
    
    dtree = DecisionTreeClassifier(criterion="gini",max_features="auto",random_state=0)
    dt = dtree.fit(Xtrain,Ytrain)
    dtree_acc = round(dt.score(Xtrain,Ytrain)*100,2)
    print("\nAccuracy of Decision Tree Classifer:",dtree_acc)
    plot_graph(dt,Xtrain)
    
    gnd =GaussianNB()
    gnd.fit(Xtrain,Ytrain)
    gnd_acc=round(gnd.score(Xtrain,Ytrain)*100,2)
    print("\nAccuracy of Gaussian Naive Bayes:",gnd_acc)
    
    sgd = SGDClassifier()
    sgd.fit(Xtrain,Ytrain)
    sgd_acc=round(sgd.score(Xtrain,Ytrain)*100,2)
    print("\nAccuracy of Stochastic Gradient Classifier:",sgd_acc)
    
    rnd_forest = RandomForestClassifier(n_estimators=10)
    rnd = rnd_forest.fit(Xtrain,Ytrain)
    rnd_forest_acc=round(rnd.score(Xtrain,Ytrain)*100,2)
    print("\nAcuracy of Random Forest:",rnd_forest_acc)
    
    svc = LinearSVC()
    svm = svc.fit(Xtrain,Ytrain)
    svc_acc = round(svm.score(Xtrain,Ytrain)*100,2)
    print("\nAccuracy of Linear SVC:",svc_acc)
    
    kneighbors = KNeighborsClassifier(n_neighbors=3)
    kn = kneighbors.fit(Xtrain,Ytrain)
    kneighbors_acc = round(kn.score(Xtrain,Ytrain)*100,2)
    print("\nAcuracy of K Nearest Neighbors Classifier:",kneighbors_acc)
    
    perceptron = Perceptron()
    perceptron.fit(Xtrain,Ytrain)
    perceptron_acc= round(perceptron.score(Xtrain,Ytrain)*100,2)
    print("\nAccuracy of Perceptron:",perceptron_acc)
    
    
    Model_Name = ["Logistic Regression","Decision Tree","Naive Bayes","Stochastic Gradient","Random Forest","Support Vector Machine","K Nearest Neighbor","Perceptron"]
    Model_Score = [lgreg_acc,dtree_acc,gnd_acc,sgd_acc,rnd_forest_acc,svc_acc,kneighbors_acc,perceptron_acc]
    AutoML_result = pd.DataFrame({"Model Name":Model_Name,"Model Score":Model_Score})
    AutoML_sort = AutoML_result.sort_values(by="Model Score",ascending=False)[0:3]
    AutoML_sort = AutoML_sort.reset_index(drop=True)
    print("\nTop 3 Classifiers:\n ",AutoML_sort)
    
    flg= False
           
    if AutoML_sort.loc[0,"Model Name"] == "Decision Tree":
       dt_acc = cross_val_score(dt,Xtrain,Ytrain,cv=10,scoring="accuracy")
       print("\n************************************************************")
       print("\nSelected Algorithm:",AutoML_sort.loc[0,"Model Name"])
       print("\nAccurcy:",dt_acc)
       print("\nScore Mean:",dt_acc.mean())
       print("\nScore Standard Deviation:", dt_acc.std())
       if len(Xtest.columns) == len(Xtrain.columns):
          predict = dt.predict(Xtest)
          print("Predicted Survival:",predict)
       else:
          print("Number of columns in Training Dataset and Testing Dataset are not equal")
       learning_curve_fit(AutoML_sort.loc[0,"Model Name"],dt,Xtrain,Ytrain,10)   
       feature_importance = pd.DataFrame({"Feature Name":Xtrain.columns,"Importance Score":np.round(dt.feature_importances_,2)})
       feature_importance_sort = feature_importance.sort_values(by="Importance Score",ascending=False).set_index("Feature Name")
       print(feature_importance_sort)
       feature_importance_top6 = feature_importance_sort.head(6)
       feature_importance_top6.plot.barh(title="Feature Selection:Decisiion Tree")
       print("\n************************************************************\n") 
     
    if AutoML_sort.loc[0,"Model Name"] == "K Nearest Neighbor":
       kn_acc = cross_val_score(kn,Xtrain,Ytrain,cv=10,scoring="accuracy")
       print("\n************************************************************")
       print("\nSelected Algorithm:",AutoML_sort.loc[0,"Model Name"])
       print("\nAccurcy:",kn_acc)
       print("\nScore Mean:",kn_acc.mean())
       print("\nScore Standard Deviation:", kn_acc.std())
       if len(Xtest.columns) == len(Xtrain.columns):
          predict = kn.predict(Xtest)
          print("Predicted Survival:",predict)
       else:
          print("Number of columns in Training Dataset and Testing Dataset are not equal")
       print("\nK Nearest Neighbor has no feature importance attribute")
       learning_curve_fit(AutoML_sort.loc[0,"Model Name"],kn,Xtrain,Ytrain,10)
       flg=True
       print("\n************************************************************\n") 
     
    if AutoML_sort.loc[0,"Model Name"] == "Logistic Regression":
       lr_acc = cross_val_score(lr,Xtrain,Ytrain,cv=10,scoring="accuracy")
       print("\n************************************************************")
       print("\nSelected Algorithm:",AutoML_sort.loc[0,"Model Name"])
       print("\nAccurcy:",lr_acc)
       print("\nScore Mean:",lr_acc.mean())
       print("\nScore Standard Deviation:",lr_acc.std())
       if len(Xtest.columns) == len(Xtrain.columns):
          predict = dt.predict(Xtest)
          print("Predicted Survival:",predict)
       else:
          print("Number of columns in Training Dataset and Testing Dataset are not equal")
       print("\nLogistic Regression has no feature importance attribute")
       learning_curve_fit(AutoML_sort.loc[0,"Model Name"],lr,Xtrain,Ytrain,10)
       flg=True
       print("\n************************************************************\n") 
       
    if AutoML_sort.loc[0,"Model Name"] == "Support Vector Machine":
       svm_acc = cross_val_score(svm,Xtrain,Ytrain,cv=10,scoring="accuracy")
       print("\n************************************************************")
       print("\nSelected Algorithm:",AutoML_sort.loc[0,"Model Name"])
       print("\nAccurcy:",svm_acc)
       print("\nScore Mean:",svm_acc.mean())
       print("\nScore Standard Deviation:", svm_acc.std())
       if len(Xtest.columns) == len(Xtrain.columns):
          predict = svm.predict(Xtest)
          print("Predicted Survival:",predict)
       else:
          print("Number of columns in Training Dataset and Testing Dataset are not equal")
       print("\nLinear Support Vector Machine has no feature importance attribute")
       learning_curve_fit(AutoML_sort.loc[0,"Model Name"],svm,Xtrain,Ytrain,10)
       flg=True
       print("\n************************************************************\n") 
       
    if AutoML_sort.loc[1,"Model Name"] == "Random Forest" or flg == True:
       rnd_acc = cross_val_score(rnd,Xtrain,Ytrain,cv=10,scoring="accuracy")
       print("\n************************************************************")
       print("\nSelected Algorithm:",AutoML_sort.loc[0,"Model Name"])
       print("\nAccurcy:",rnd_acc)
       print("\nScore Mean:",rnd_acc.mean())
       print("\nScore Standard Deviation:",rnd_acc.std())
       if len(Xtest.columns) == len(Xtrain.columns):
          predict = rnd.predict(Xtest)
          print("Predicted Survival:",predict)
       else:
          print("Number of columns in Training Dataset and Testing Dataset are not equal")
       learning_curve_fit(AutoML_sort.loc[1,"Model Name"],rnd,Xtrain,Ytrain,10)
       feature_importance = pd.DataFrame({"Feature Name":Xtrain.columns,"Importance Score":np.round(rnd.feature_importances_,2)})
       feature_importance_sort = feature_importance.sort_values(by="Importance Score",ascending=False).set_index("Feature Name")
       print(feature_importance_sort)
       feature_importance_top6 = feature_importance_sort.head(6)
       feature_importance_top6.plot.barh(title="Feature Selection:Random Forest")
       print("\n************************************************************\n")  
        
    return AutoML_sort


    
        
train_df,test_df=load_data(folder_path,file,5)

X_train,Y_train,X_test=automatic_EDA(train_df,test_df,"Survived","PassengerId",["PassengerId","Ticket","Name","Cabin"])
AutoML = Auto_ML(X_train,Y_train,X_test)

