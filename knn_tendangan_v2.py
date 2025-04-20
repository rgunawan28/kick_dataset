import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#ambil dataset
kick = pd.read_csv("seg_dataset.csv")
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
x = scaled_data
#membagi data untuk Training dan Test
x_training_data,x_test_data,y_training_data,y_test_data=train_test_split(x,y,test_size=0.3)
#set KNN = 1,3, atau 5
model = KNeighborsClassifier(n_neighbors =5)
model.fit(x_training_data,y_training_data)
prediction = model.predict(x_test_data)
# tanpilkan test data dan hasil prediksi
print(x_test_data)
print(prediction)
#cetak klasisfikasi report
print(classification_report(y_test_data,prediction))
#cetak confusion matrix
print(confusion_matrix(y_test_data,prediction))
#Mencari nilai k value yang optimum
error_rates = []
for i in np.arange(1,24):
    new_model = KNeighborsClassifier(n_neighbors = i)
    new_model.fit(x_training_data,y_training_data)
    new_predictions = new_model.predict(x_test_data)
    error_rates.append(np.mean(new_predictions != y_test_data))
plt.plot(error_rates)
plt.show()
