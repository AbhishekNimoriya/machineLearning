import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabities = datasets.load_diabetes()

# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

# print(diabities.target)

diabities_x= diabities.data

diabities_x_train = diabities_x[:-30]
diabities_x_test = diabities_x[-30:]

diabities_y_train = diabities.target[:-30]
diabities_y_test = diabities.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabities_x_train, diabities_y_train)

diabities_y_predict = model.predict(diabities_x_test)

# print("Mean squard error is : ", mean_squared_error(diabities_y_test, diabities_y_predict)) // Meansquaderror - 3053

print("Mean squard error is : ", mean_squared_error(diabities_y_test, diabities_y_predict))

print("Wieghts : ", model.coef_)
print("Intercepts : ", model.intercept_)


# Plotting graph

# plt.scatter(diabities_x_test, diabities_y_test)
# plt.plot(diabities_x_test, diabities_y_predict)

# plt.show()
