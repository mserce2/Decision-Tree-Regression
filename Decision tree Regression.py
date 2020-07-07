
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Tribun_Oran.csv",sep=";",header=None)
print(df)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%% decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(x,y)

print(tree_reg.predict([[11.5]]))
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=tree_reg.predict(x_)

#%% visualize
plt.scatter(x,y,edgecolors="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()