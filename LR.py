import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]

x, y = np.array(x), np.array(y)
print(y)

model = LinearRegression().fit(x, y)

print('model score : ', model.score(x, y))

z = np.array([[80, 50],[30, 10]])
output = model.predict(z)

print(output)

pickle.dump(model,open('model.pkl','wb'))