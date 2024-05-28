# trainieren einer Neuronal netzte, um die doppelt einer Zahl zu finden

from keras import *         # alle Funktionen importieren
import numpy as np

model = Sequential()    # ein Model erstellen mit der aufruf der funktion sequential

# input layer hinzufügen, wo die Neurone verknupft sind
model.add(layers.Dense(units=3, input_shape=[1]))


# zwischen Layers hinzufügen
model.add(layers.Dense(units=64))

# ouput layer mit einer einzigen Neuron
model.add(layers.Dense(units=1))

input_data = [1, 2, 3, 4, 5]
output_data = [2, 4, 6, 8, 10]   

# convert the data to the numpy arrays
input_array = np.array(input_data).reshape(-1, 1)
output_array = np.array(output_data)

# on compile notre model et lui indiquer les fonctions que l'on veut utilser pour qu'il puisse s'ameliorer, s'optimiser 
# model compilieren und verschiedene Functionen hinzufügen
model.compile(loss='mean_squared_error', optimizer='adam')

# hier trainieren wir unser model 
model.fit(x=input_array, y=output_array, epochs=2000)

# vorhersage schleife
while True:
    user_input = int(input("Geben Sie eine Zahl ein: "))
    user_input_array = np.array([[user_input]])
    prediction = model.predict(user_input_array)
    print(f"Vorhersage: {prediction[0][0]}")

