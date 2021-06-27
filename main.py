from keras.models import Sequential  # model create
from keras.layers import Dense  # layers NN
from sklearn.model_selection import train_test_split  # data division
import numpy

# fix random seed for reproducibility
# numpy.random.seed(7)
# load diabetes dataset
dataset = numpy.loadtxt("diabetes.csv", delimiter=",")
# print(dataset)
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]  # independent variable
Y = dataset[:, 8]  # dependent variable   Y~X
# print(X)
# print(Y)
Xtr, Xts, ytr, yts = train_test_split(X, Y, test_size=0.1, random_state=10)  # 90% train 10% test
# # create model

model = Sequential()  # linear model
model.add(Dense(12, input_dim=8, activation='relu'))  # input layer
model.add(Dense(8, activation='relu'))  # interm layer
model.add(Dense(4, activation='relu'))  # interm layer
model.add(Dense(1, activation='sigmoid'))  # output layer
# # Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
model.fit(Xtr, ytr, epochs=200, batch_size=50)  # iteration
# # evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# print(model.predict(Xts))
from ann_visualizer.visualize import ann_viz

ann_viz(model, title="My first neural network")

# dataset = numpy.loadtxt("cars.csv", delimiter=",")
# print(dataset)
# # split into input (X) and output (Y) variables
# X = dataset[:, 0:5]  # independent variable
# Y = dataset[:, 5]  # dependent variable   Y~X
# # print(X)
# # print(Y)
# Xtr, Xts, ytr, yts = train_test_split(X, Y, test_size=0.1, random_state=10)  # 90% train 10% test
# # # create model
#
# model = Sequential()  # linear model
# model.add(Dense(12, input_dim=5, activation='relu'))  # input layer
# model.add(Dense(8, activation='relu'))  # interm layer
# model.add(Dense(1, activation='sigmoid'))  # output layer
# # # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # # Fit the model
# model.fit(Xtr, ytr, epochs=10, batch_size=10)  # iteration
# # # evaluate the model
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# print(model.predict(Xts))
