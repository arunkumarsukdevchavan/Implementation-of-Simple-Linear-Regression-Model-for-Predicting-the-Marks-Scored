# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and sklearn
2. Calculate the values for the  training data set
3. Calculate the values for the  test data set
4. Plot the graph for both the data sets and calculate for MAE, MSE and RMSE

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ARUN KUMAR SUKDEV CHAVAN
RegisterNumber:  212222230013

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/student_scores.csv')
#Displying the contents in datafile
df.head()

df.tail()

#Segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,-1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying the predicted values
Y_pred

#displaying the actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="green")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE= ",mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
#### df.head()
![Screenshot 2023-03-29 141212](https://user-images.githubusercontent.com/115707860/228477967-5b670ef7-76ee-4a5f-94d7-c605b58d067d.png)


#### df.tail()
![Screenshot 2023-03-29 141344](https://user-images.githubusercontent.com/115707860/228478274-989aafdb-82cf-4257-9a3a-0317ced42178.png)
#### X
![Screenshot 2023-03-29 141416](https://user-images.githubusercontent.com/115707860/228478461-0a185d4f-afb3-47b0-9052-666f7b3bf265.png)

#### Y
![Screenshot 2023-03-29 141503](https://user-images.githubusercontent.com/115707860/228478618-f7f6a0a2-36d7-4dc2-8db9-08ecdf846ef8.png)

#### PREDICTED Y VALUES
![Screenshot 2023-03-29 141503](https://user-images.githubusercontent.com/115707860/228479078-a394d358-4d5e-479c-bc4d-716a010081a8.png)

#### ACTUAL Y VALUES
![Screenshot 2023-03-29 142954](https://user-images.githubusercontent.com/115707860/228482752-cf040c75-ae8b-4f0d-8ea2-476424eb079e.png)

#### GRAPH FOR TRAINING DATA
![Screenshot 2023-04-04 185208](https://user-images.githubusercontent.com/115707860/229806294-0e5783ed-1463-48c1-b93e-460ce785e7af.png)

#### GRAPH FOR TEST DATA
![Screenshot 2023-04-04 185256](https://user-images.githubusercontent.com/115707860/229806518-03370174-b337-4a62-94ac-90d97c64cc4d.png)

### MEAN SQUARE ERROR, MEAN ABSOLUTE ERROR AND RMSE
![Screenshot 2023-04-04 185433](https://user-images.githubusercontent.com/115707860/229807013-9a2db0a2-04c4-466f-af27-6586dfd53642.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
