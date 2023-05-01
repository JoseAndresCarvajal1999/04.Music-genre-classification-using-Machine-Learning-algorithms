import numpy as np
from scipy.special import binom

def pac_bound(vc, epsilon, delta):
    " Redes neuronales, regresiones lineales, regresiones polinomicas"
    return (1/epsilon*(8*vc*np.log(13/epsilon) + 4*np.log(2/delta)))

def pac_bound_arboles(num_features, depth, epsilon, delta):
    return (np.log(2) / (2 * epsilon ** 2 )) * ((2 ** depth - 1) * (1 + np.log2(num_features)) + 1 + np.log(delta ** -1))

#Bajas Dimensiones 

print('------------------- Regresiones lineales --------------------') 
#2 Dimensiones 
vc = 2+1
for delta, epsilon in [(0.02, 0.05), (0.07, 0.10), (0.10, 0.15)]:
    n = pac_bound(vc, delta, epsilon)
    print('delta = {:.2f}, epsilon = {:.2f}, m >= {:.2f}'.
          format(delta, epsilon, n))

print('-----------------------------------------------------')
#3 Dimensiones 
vc = 3 +1
for delta, epsilon in [(0.02, 0.05), (0.07, 0.10), (0.10, 0.15)]:
    n = pac_bound(vc, delta, epsilon)
    print('delta = {:.2f}, epsilon = {:.2f}, m >= {:.2f}'.
          format(delta, epsilon, n))

print('-----------------------------------------------------')
#58 Dimensiones 
vc = 58 +1
for delta, epsilon in [(0.02, 0.05), (0.07, 0.10), (0.10, 0.15)]:
    n = pac_bound(vc, delta, epsilon)
    print('delta = {:.2f}, epsilon = {:.2f}, m >= {:.2f}'.
          format(delta, epsilon, n))

        

print('------------------- Arboles --------------------') 
depth = 19
#2 Dimensiones 
num_features = 2
for delta, epsilon in [(0.02, 0.05), (0.07, 0.10), (0.10, 0.15)]:
    n = pac_bound_arboles(num_features, depth, delta, epsilon)
    print('delta = {:.2f}, epsilon = {:.2f}, m >= {:.2f}'.
          format(delta, epsilon, n))

print('-----------------------------------------------------')
#3 Dimensiones 
num_features = 3
for delta, epsilon in [(0.02, 0.05), (0.07, 0.10), (0.10, 0.15)]:
    n = pac_bound_arboles(num_features, depth, delta, epsilon)
    print('delta = {:.2f}, epsilon = {:.2f}, m >= {:.2f}'.
          format(delta, epsilon, n))

#58 Dimensiones 
num_features = 58
for delta, epsilon in [(0.02, 0.05), (0.07, 0.10), (0.10, 0.15)]:
    n = pac_bound_arboles(num_features, depth, delta, epsilon)
    print('delta = {:.2f}, epsilon = {:.2f}, m >= {:.2f}'.
          format(delta, epsilon, n))

print('------------------- Redes neuronales --------------------') 
#2 dimensiones 
vc = 2850
for delta, epsilon in [(0.02, 0.05), (0.07, 0.10), (0.10, 0.15)]:
    n = pac_bound(vc, delta, epsilon)
    print('delta = {:.2f}, epsilon = {:.2f}, m >= {:.2f}'.
          format(delta, epsilon, n))
print('-----------------------------------------------------')
#3 Dimensiones 
vc = 12300
for delta, epsilon in [(0.02, 0.05), (0.07, 0.10), (0.10, 0.15)]:
    n = pac_bound(vc, delta, epsilon)
    print('delta = {:.2f}, epsilon = {:.2f}, m >= {:.2f}'.
          format(delta, epsilon, n))
print('-----------------------------------------------------')
#58 Dimensiones 
vc = 146300
for delta, epsilon in [(0.02, 0.05), (0.07, 0.10), (0.10, 0.15)]:
    n = pac_bound(vc, delta, epsilon)
    print('delta = {:.2f}, epsilon = {:.2f}, m >= {:.2f}'.
          format(delta, epsilon, n))
