# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sm

# Step 1: Prepare your data
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
#################################################################
# Start Algorithm here
#################################################################
print("Random Forest Algorithm")
print("========================")
# Assuming you have your feature data in X and label data in y
# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
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
