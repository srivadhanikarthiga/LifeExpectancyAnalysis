#Importing Required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model                                            
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 

#Load the dataset
df=pd.read_csv("Life_expectancy.csv")
print(df)

print("Head of the dataset")
print(df.head())

print("Tail of the dataset")
print(df.tail())
print("Shape of the dataset")
print(df.shape)
print("Information of the dataset")
print(df.info())
#Finding the Count of the dataset
print("Count the values:",df.count())

#Finding the Missing values in the dataset
print("missing values in the dataset:")
print(df.isnull().sum())

#Finding the Duplicated Items of the dataset
d=df[df.duplicated()]
print("Duplicate entries:")
print(d)

#descriptive statistics
print("Mean=\n",df.mean())
print("Median=\n",df.median())
print("Variance=\n",df.var())
print("Standard deviation=\n",df.std())
print("Maximum value=\n",df.max())
print("Minimum value=\n",df.min())

#Finding the Interquartile range of the dataset
print("Interquartile=",df.quantile())

#Aggregate functions
x=df.aggregate(["sum"])
print(x)
y=df.aggregate(["max"])
print(y)
z=df.aggregate(["mean"])
print(z)
s=df.aggregate(["sem"])
print(s)
p=df.aggregate(["var"])
print(p)
q=df.aggregate(["prod"])
print(q)


# descriptive statistics for Grouped data
df1=df.groupby(['Lifeexpectancy'])
print(df1.first())
print("Mean=\n",df1['Lifeexpectancy'].mean())
print("Median=\n",df1['Lifeexpectancy'].median())
print("Variance=\n",df1['Lifeexpectancy'].var())
print("Standard deviation=\n",df1['Lifeexpectancy'].std())
print("Maximum value=\n",df1['Lifeexpectancy'].max())
print("Minimum value=\n",df1['Lifeexpectancy'].min())


#Skewness
print(df.skew())

#Kurtosis
print(df.kurtosis())

#Visualization of data

import seaborn as sns
fig=plt.figure()
ax=plt.axes(projection='3d')
x=df['Lifeexpectancy']
y=df['Avg_Life_Expec']
z=df['Age_Adj_Death_Rate']
ax.plot3D(x,y,z,'purple')
ax.set_title('covid-19 dataset')
plt.show()

plt.plot(df.Year,df.Lifeexpectancy)
plt.title("Year vs Life expectancy")
plt.xlabel("Year")
plt.ylabel("Life expectancy ")
plt.show()

sns.pairplot(data=df)
plt.show()


plt.hist(df.Lifeexpectancy,bins=30)
plt.xlabel("Life Expectancy")
plt.show()

df.plot(kind='box',subplots=True,layout=(5,3),figsize=(12,12))
plt.show()

f,ax=plt.subplots(figsize=(10,6))
x=df['Lifeexpectancy']
ax=sns.kdeplot(x,shade=True,color='r')
plt.show()

sns.heatmap(df.corr())
plt.show()

plt.scatter(df.Year,df.Lifeexpectancy)
plt.title("Year vs Life expectancy")
plt.xlabel("Year")
plt.ylabel("Life expectancy")
plt.show()

X =df['Year']
y =df['Lifeexpectancy']
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.59, test_size = 0.41, random_state =110)

print(X.shape)
print(y.shape)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)

b = regr.coef_
b


a = regr.intercept_
a

plt.scatter(X_train, y_train)
plt.plot(X_train,a + b*X_train, 'r')
plt.show()

y_pred = regr.predict(X_test)

res = (y_test - y_pred)
print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred))
print('Mean Absolute Error: %.2f'% mean_absolute_error(y_test, y_pred))

X=df[['Year','Avg_Life_Expec','Age_Adj_Death_Rate']]
Y=df['Lifeexpectancy']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=101)
reg=linear_model.LinearRegression()
reg.fit(X_train,Y_train)
Y_predict=reg.predict(X_test)
print('Coefficients:',reg.coef_)

print('Variance score:{}'.format(reg.score(X_test,Y_test)))

Variance score:0.6461364638987

from sklearn.metrics import r2_score
print('r^2:',r2_score(Y_test,Y_predict))

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,Y_predict)
rmse=np.sqrt(mse)
print('RMSE:',rmse)
