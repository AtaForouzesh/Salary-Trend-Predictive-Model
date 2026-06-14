import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

data = pd.read_csv("Regressions/My Training Tasks/Salary_Data.csv")

x = data.iloc[:, 0].values
y = data.iloc[:, -1].values

x = np.array(x)
y = np.array(y)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=123)

lireg = LinearRegression()
lireg.fit(X_train, y_train)

y_ = lireg.predict(X_test)
r2 = r2_score(y_test, y_)
print(r2)

plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, y_, color="blue")
plt.title("Salary based on Years Of Experience")
plt.xlabel("Experience Level")
plt.ylabel("Salaries")
plt.show()
