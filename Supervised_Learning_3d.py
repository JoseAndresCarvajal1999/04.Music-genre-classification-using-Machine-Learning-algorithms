import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 

pca = np.array(pd.read_csv('PCA.csv'))

file = pd.read_csv('features_30_sec.csv')
columns = file.columns
list_names = []
for name in columns: 
    list_names.append(name)

list_names.remove('filename')
list_names.remove('label')
input_variables = file[list_names]
ouput_variable = np.array(file['label']).reshape(-1,1)
ouput_variable = ouput_variable[1:]
#Se decodifican las etiquetas de las categorias  
enc = OrdinalEncoder()
enc.fit(ouput_variable)
Var_categoricas_enc = enc.transform(ouput_variable)
pca= pca[:-1,0:3]
scaler = MinMaxScaler()
pca = scaler.fit_transform(pca)

print('------------- PCA------------------------')

x_train,x_rest, y_train, y_rest = train_test_split(pca,Var_categoricas_enc, test_size=0.4)
x_val, x_test, y_val, y_test = train_test_split(x_rest,y_rest, test_size = 0.5)


#------------  Maquina de soporte vecotiral -------------------------
#--------Entrenar ------------------------
regresor = svm.SVC()
regresor.fit(x_train, y_train)
#-------- Validar -------------------------
y_pred1 = regresor.predict(x_val)
#print('Error maximo : ',accuracy_score(y_pred1,y_val))
#print('Procenta medio de error : ',mean_absolute_percentage_error(y_pred1,y_val))
#print('MSE: ',mean_squared_error(y_pred1,y_val))

#-------- Testear ------------------------------------------
print('---------------  Testeo Maquina de soporte vectorial ----------------')
y_pred1 = regresor.predict(x_test)
print(y_pred1)
print('Exactitud : ',accuracy_score(y_pred1,y_test))
print('Precision de la exactitud: ',precision_score(y_pred1,y_test, average='macro'))

#plt.plot(y_pred1, 'or')
#plt.show()
#plt.plot(y_test, 'ob')
#plt.show()

#------------  Red Neuronal -------------------------
#--------Entrenar ------------------------
regresor = MLPClassifier(hidden_layer_sizes = (100,))
regresor.fit(x_train, y_train)
#-------- Validar -------------------------
y_pred1 = regresor.predict(x_val)
#print('Error maximo : ',accuracy_score(y_pred1,y_val))
#print('Procenta medio de error : ',mean_absolute_percentage_error(y_pred1,y_val))
#print('MSE: ',mean_squared_error(y_pred1,y_val))

#-------- Testear ------------------------------------------
print('---------------  Testeo red neuronal ----------------')
y_pred1 = regresor.predict(x_test)
print(y_pred1)
print('Exactitud : ',accuracy_score(y_pred1,y_test))
print('Precision de la exactitud: ',precision_score(y_pred1,y_test, average='macro'))
x = np.arange(0,20,1)

plt.plot(x,y_pred1[0:20], 'r')
plt.plot(x,y_test[0:20], 'ob')
plt.xlabel('Muestra')
plt.xlim((0,10))
plt.ylabel('Clasificación en generos musicales')
plt.show()

#------------  Arbol de decision -------------------------
#--------Entrenar ------------------------
regresor = DecisionTreeClassifier()
regresor.fit(x_train, y_train)
#-------- Validar -------------------------
y_pred1 = regresor.predict(x_val)
#print('Error maximo : ',accuracy_score(y_pred1,y_val))
#print('Procenta medio de error : ',mean_absolute_percentage_error(y_pred1,y_val))
#print('MSE: ',mean_squared_error(y_pred1,y_val))

#-------- Testear ------------------------------------------
print('---------------  Testeo arbol de decision ----------------')
y_pred1 = regresor.predict(x_test)
print(y_pred1)
print('Exactitud : ',accuracy_score(y_pred1,y_test))
print('Precision de la exactitud: ',precision_score(y_pred1,y_test, average='macro'))

x = np.arange(0,20,1)

#print(x)
#plt.plot(x,y_pred1[0:20], 'r')
#plt.plot(x,y_test[0:20], 'ob')
#plt.xlabel('Muestra')
#plt.xlim((0,10))
#plt.ylabel('Clasificación en generos musicales')
#plt.show()



