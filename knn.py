import numpy as np
import tensorflow as tf
from sklearn import neighbors
from sklearn.metrics import accuracy_score
#import cnn
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train[:60000]
y_train = y_train[:60000]
x_test = x_test[:1000]
y_test = y_test[:1000]

print(x_train.shape)
print(x_test.shape)

# reshaping because KNN expect a vector not a matrix
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

kVals = np.arange(3,11,2)
accuracies = []
for k in kVals:
  tic = time.perf_counter()

  model = neighbors.KNeighborsClassifier(n_neighbors = k, weights="uniform")
  model.fit(x_train, y_train)

  pred = model.predict(x_test)
  tac = time.perf_counter()

  acc = accuracy_score(y_test, pred)
  accuracies.append(acc)
  print("K = "+str(k)+"; Accuracy: "+str(acc)+f"; time = {tac-tic:0.4f} seconds")



