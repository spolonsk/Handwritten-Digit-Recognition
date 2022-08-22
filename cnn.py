from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras.layers.convolutional import Conv2D # to add convolutional layers
from keras.layers.convolutional import MaxPooling2D # to add pooling layers
from keras.layers import Flatten
from keras.datasets import mnist

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:60000]
y_train = y_train[:60000]
x_test = x_test[:1000]
y_test = y_test[:1000]

print(x_train.shape)
print(x_test.shape)

X_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# normalize the pixel values to be between 0 and 1

X_train = X_train / 255 # normalize training data
X_test = X_test / 255 # normalize test data

# convert the target variable into binary categories

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1] # number of categories (0 jusque 9) 10 catégories/10 classes

def convolutional_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(Flatten())
    
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# calling the function to create the model
model = convolutional_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# evaluate the model
print("evaluation en cours")
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy : {} \Error {}".format(scores[1], 100-scores[1]*100))
print("after evaluation")

model_json = model.to_json()
with open("model.json", "w",) as json_file:
    json_file.write(model_json)

model.save("final_model.h5")

