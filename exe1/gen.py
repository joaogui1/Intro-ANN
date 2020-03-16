import json as js
import random as rnd
normal_a = [[-1, -1, -1, 1, -1, -1, -1], [-1, -1, 1, -1, 1, -1, -1], [-1, 1, 1, -1, 1, 1, -1], [-1, 1, -1, -1, -1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [ -1, 1, -1, -1, -1, 1, -1], [ -1, 1, -1, -1, -1, 1, -1]] 
inv_a = normal_a[::-1]
 
for k in range(40):
    with open(str(k) + ".in", 'w') as inp:
        aux = list(normal_a) 
        for i in range(7):
            for j in range(7):
                change = rnd.randint(1, 100)
                if change <= 3: 
                    aux[i][j] = -1*aux[i][j]
        js.dump([aux, 1], inp)                     
    
for k in range(40, 80):
    with open(str(k) + ".in", 'w') as inp:
        aux = list(inv_a) 
        for i in range(7):
            for j in range(7):
                change = rnd.randint(1, 100)
                if change <= 3: 
                    aux[i][j] = -1*aux[i][j]
        js.dump([aux, -1], inp) 
              

for k in range(80, 90):
    with open(str(k) + ".in", 'w') as inp:
        aux = list(normal_a) 
        for i in range(7):
            for j in range(7):
                change = rnd.randint(1, 100)
                if change <= 3: 
                    aux[i][j] = -1*aux[i][j]
        js.dump([aux, 1], inp) 

for k in range(90, 100):
    with open(str(k) + ".in", 'w') as inp:
        aux = list(inv_a) 
        for i in range(7):
            for j in range(7):
                change = rnd.randint(1, 100)
                if change <= 3: 
                    aux[i][j] = -1*aux[i][j]
        js.dump([aux, -1], inp) 
   
