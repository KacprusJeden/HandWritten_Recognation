import cv2 as cv
import numpy as np
import keras as k
import matplotlib.pyplot as plt
import tensorflow as tf

# ładowanie próbek obrazów do walidacjii uczenia
mnist = k.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(y_test)

#normalizacja obrazow do pixeli z zakresu [0;1] po osi Y
x_train = k.utils.normalize(x_train, axis=1)
x_test = k.utils.normalize(x_test, axis=1)

# tworzymy model sekwencyjny
model = k.models.Sequential()
model.add(k.layers.Flatten(input_shape=(28,28))) # wejściowa warstwa płaska - obraz 28x28
model.add(k.layers.Dense(units=128, activation=tf.nn.relu)) # warstwa głęboka, 128 neuronów
model.add(k.layers.Dense(units=128, activation=tf.nn.relu)) # warstwa głęboka, 128 neuronów
model.add(k.layers.Dense(units=10, activation=tf.nn.softmax)) #warstwa wyjściowa, 15 neuronów, 10 to jest wymagane minimum

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # kompilowanie modelu

model.fit(x_train, y_train, epochs=3, batch_size=32) # uczenie modelu
model.save('digits.model') # zapisanie modelu

loss, accuracy = model.evaluate(x_test, y_test) #wyciągnięcie z modelu straty i dokładności
print(accuracy)
print(loss)

# test of program
# każdy obraz będzie czarno-biały, wyciągamy predykcję, czyli w naszym przypadku przewidujemy, jaka liczba może być na obrazie
for i in range(1,51):
    img = cv.imread(f'28_28/{i}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'Probably the number is: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()