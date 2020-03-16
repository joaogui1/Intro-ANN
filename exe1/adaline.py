import json as js
import numpy as np
import random as rd
import matplotlib.pyplot as mpl

class adaline(object):
    def __init__(self, eta = 0.01, weights = []):
        self.eta = eta
        #casos não sejam dados pesos para o neurônio ele gera pesos iniciais aleatórios
        if (weights == []):
            self.weights = np.random.rand(50)
        else:
            self.weights = weights

    #metodo que calcula o label dado o valor do produto escalar entre os pesos da rede e a entrada
    def label(self, res):
        if(res >= 0.0):
            return 1
        else:
            return -1

    #Metodo que treina a ADALINE
    def train(self, threshold=1e-4):
        #counter guarda o número de epochs até a convergências
        counter = 0
        sqerror = threshold + 1

        #enquanto o erro quadratico medio não for menor que o limiar
        while(sqerror > threshold):
            counter += 1
            sqerror = 0.0

            for image in range(0, 80):
                #carrega cada imagem de treino
                with open(str(image) + ".in", 'r') as inp:
                    #formata a entrada como um array de numpy para facilitar a manipulação
                    data = js.load(inp)
                    data[0] = np.array(data[0], dtype=np.float128)
                    #Concatena-se um ao final da linha para simplicar os cálculos com o bias
                    data[0] = np.append(np.reshape(data[0], 49), [1.0])

                    #Rótulo obtido pela rede
                    lab = self.label(np.dot(data[0], self.weights))

                    #cálculo do erro e atualização do erro quadratico medio
                    error = (data[1] - lab)
                    sqerror += (error**2)/80.0

                    #calculo do gradiente para o gradient descent
                    gradient = -error*data[0]

                    #gradient descent
                    self.weights -= self.eta*gradient
        return counter

    #método que testa de o neurônio realmente aprendeu a classificar os dados
    def test(self):
        error = 0 #variavel que guarda o percentual de erros da ADALINE
        for i in range(80, 100):

            with open(str(i) + ".in", 'r') as inp:
                #prepara os dados como no treinamento
                data = js.load(inp)
                data[0] = np.array(data[0], dtype=np.float128)
                data[0] = np.append(np.reshape(data[0], 49), [1.0])

                lab = self.label(np.dot(data[0], self.weights))
                if(lab != data[1]):
                   error += 1.0

        return 5.0*error


epochs = []
results = []

#gera os dados para os histogramas
for testes in range(100):
    rd.seed(testes)
    ANN = adaline()
    epochs.append(ANN.train())
    results.append(ANN.test())

bins = np.arange(0, 100, 5)
mpl.xlim(0, 100)
mpl.hist(results, bins = bins)
mpl.title("Test error with different initial weights histogram")
mpl.xlabel("Error")
mpl.ylabel("Count")
mpl.show()

mpl.hist(epochs)
mpl.title("Number of epochs needed to converge")
mpl.xlabel("Epochs")
mpl.ylabel("Count")
mpl.show()
