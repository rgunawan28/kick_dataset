# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sm

# Step 1: Prepare data
#ambil dataset
kick = pd.read_csv("seg_dataset_RS.csv")
print(kick.head())
print(kick)
print(kick.columns)
#ambil target label
y=kick['Label']
print (y)
# Standarize agar scalanya nya hampir sama pre processing
scaler = StandardScaler()
scaler.fit(kick.drop('Label',axis=1))
scaled_features = scaler.transform(kick.drop('Label',axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = kick.drop('Label',axis=1).columns)
X = scaled_data
# Assuming you have your feature data in X and label data in y
# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#################################################################
# Start Algorithm here
#################################################################
#print("Linier Regression  Algorithm")
#print("========================")
#lr = LogisticRegression()
#lr.fit(X_train,y_train)
#lr_score = lr.score(X_test,y_test)
#print ('Score=',lr_score)       
print("Random Forest Algorithm")
print("========================")
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(X_test)
print(predictions)
Score = model.score(X_test,y_test)
print ('Score=',Score)
cm = confusion_matrix(y_test,predictions)
print(cm)
plt.figure(figsize=(10,7))
sm.heatmap(cm,annot=True)
plt.xlabel('Prediksi')
plt.ylabel('Kenyataan')
plt.show()
print ("Random Forest END Here..\n\n")
print ("Start Algorithm SVM")
# Step 3: Create an instance of the SVM classifier
#clf = SVC(kernel='linear')
clf = SVC()
# Step 4: Train the SVM classifier
clf.fit(X_train, y_train)
# Step 5: Make predictions with the trained model
predictions = clf.predict(X_test)
# tanpilkan test data dan hasil prediksi
print(X_test)
print(predictions)
# Step 6: Evaluate the performance of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
#####################################################
# Using KFold
#####################################################
kf = KFold(n_splits = 10)
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    #return model.score(X_test,y_test)
    return score
#----------------------------------------------------
#test function
score_svm =[]
score_rf=[]
score_knn3=[]
score_knn5=[]
i=0
for train_index, test_index in kf.split(X):
    i=i+1
    print (" putaran ke %d" % i)
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    #print(X.iloc[1])
    #print ("train index......")
    #X_train = X.take(train_index)
    #print(X_train)
    #X_train, X_test, y_train, y_test = X.take(train_index),X.take(test_index), y.take(train_index),y.take(test_index)
    #print(X_train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    score_svm.append(get_score(SVC(),X_train, X_test, y_train, y_test))
    score_rf.append(get_score(RandomForestClassifier(n_estimators=50),X_train, X_test, y_train, y_test))
    score_knn3.append(get_score(KNeighborsClassifier(n_neighbors =3),X_train, X_test, y_train, y_test))
    score_knn5.append(get_score(KNeighborsClassifier(n_neighbors =5),X_train, X_test, y_train, y_test))
print("SVM ",score_svm)
print("RF ",score_rf)
print("KNN3 ",score_knn3)
print("KNN5 ",score_knn5)
print('\nSVM accuracy: %.3f +/- %.3f' % (np.mean(score_svm), np.std(score_svm)))
print('\nRF accuracy: %.3f +/- %.3f' % (np.mean(score_rf), np.std(score_rf)))
print('\nKNN3 accuracy: %.3f +/- %.3f' % (np.mean(score_knn3), np.std(score_knn3)))
print('\nKNN5 accuracy: %.3f +/- %.3f' % (np.mean(score_knn5), np.std(score_knn5)))
