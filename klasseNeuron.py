# Importieren der notwendigen Bibliotheken
# hier ist Numpy

import numpy as np

# Definition der KLasse Neuron
# Diese Klasse representiert ein einfaches künstliches Neuron,
# welches einen Inputvektor und eine Aktivierungsfunktion
# übergeben bekommt und die resultierende Aktivierung als Ergebnis zurückgibt

class Neuron:
    """
    Argumente:  dim_input (int): Dimension des Inputvektor x
                act_func:        Aktivierungsfunktion des Neurons

    Attribute:  w (ndarray) werte mit denen, der Input x gewichtet wird
                b (float)   Werte des Bias (Offset)

    """
    # Konstruktor
    def __init__(self, dim_input, act_func):
        
        # zufällige initialisierung der Gewichte (w und b)
        self.w = np.random.standard_normal(dim_input)
        self.b = np.random.standard_normal(1)

        # Aktivierungsfunktion
        self.act_func = act_func

    # Die Funktion "forward" prozessiert den Input "0027x"0027
    # durch das Neuron unter verwendung der Aktuellen
    # Dewichte sowie Anwendung der Aktivierungsfunktion

    def forward(self, x):

        """
        Argumente:  x (ndarray)  --> Größe:  (1, num_inputs)

        Rückgabewerte:  y (ndarray) --> Größe:  (1, layer_size)  
        """

        # Durchführung der Gewichtung und Addition des Bias
        z = np.dot(x, self.w) + self.b

        # Der gewichtete Input (net) wird der Aktivierungsfunktion übergeben und
        # als Ergebnis zurückgegeben.

        y = self.act_func(z)


        return y
    
# Beispielaktivierungsfunktion: Sigmoid-Funktion
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Beispielcode zur Verwendung des Neurons
if __name__ == "__main__":
    # Dimension des Inputvektors
    dim_input = 3
    # Aktivierungsfunktion
    act_func = sigmoid
    # Neuron erstellen
    neuron = Neuron(dim_input, act_func)
    # Beispiel-Eingabevektor
    x = np.array([0.5, -0.3, 0.2])
    # Vorwärtsdurchlauf durch das Neuron
    y = neuron.forward(x)
    print("Ergebnis des Neurons:", y)