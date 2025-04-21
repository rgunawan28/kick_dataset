Description of Shorinji Kempo Kicks dataset

The dataset contains kicks in kempo martial arts for the introduction of kick types.
The kicks stored are 3 types of kicks, namely front kick, side kick and round house kick.
Data contains gyroscope and accelerometer from MPU6050 sensor sent via WIFI to server collection using UDP protocol.
Data format has been pre-processed so that there are 200 samples for each kick, both for accelerometer sensor and gyroscope sensor. Because each has three axes, there will be 200x3+200x3 = 1200 data for each sensor.
If using 3 sensors, there will be 1200x3 = 3600 data for each kick. The data sequence is as follows: label, ax, ay, az,.. gx, gy, gz

[5 rows x 3601 columns]
      Label           0           1           2  ...  3596  3597  3598  3599
0   Mawashi  -49.842488  -48.933059  -47.744725  ...   -36   -36   -36   -36
1   Mawashi  -52.095161  -50.590129  -49.051993  ...     7     7     7     7
2   Mawashi  -45.888497  -44.878323  -43.868096  ...    38    38    38    38
3   Mawashi  -37.700383  -37.295380  -36.825133  ...   -32   -32   -32   -32
4   Mawashi -159.546866 -157.499815 -154.487946  ...  -105  -105  -105  -105

 Data File
1.	seg_dataset.csv   Dataset using 3 sensors
2.	seg_dataset_RSRF.csv Dataset using 2 sensor Shank and Feet
3.	seg_dataset_RLRS.csv Dataset using 2 sensor Leg and Shank
4.	seg_dataset_RS.csv Dataset using 1 sensor Shank only
5.	\randomforest_tendangan.py
6.	kFoldCross_tendangan.py
7.	knn_tendangan.py
8.	knn_tendangan_RLRS_Only.py
9.	knn_tendangan_RS_Only.py
10.	knn_tendangan_RSRF_Only.py
11.	knn_tendangan_v2.py
12.	knn_tendangan_v2_25.py
13.	test_dataframe.py
14.	test_kfold.py
15.	svm_tendangan.py
16.	svm_tendangan_v2.py
