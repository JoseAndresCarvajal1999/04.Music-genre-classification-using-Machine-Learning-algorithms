from inspect import indentsize
from numpy.core.defchararray import index
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits import mplot3d
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


file = pd.read_csv('features_30_sec.csv')
columns = file.columns
list_names = []
for name in columns: 
    list_names.append(name)

list_names.remove('filename')
list_names.remove('label')
input_variables = file[list_names]
ouput_variable = np.array(file['label']).reshape(-1,1)


#Se mapea los datos de entrada al hipercubo (0,1) 

scaler = MinMaxScaler()
Var_numericas_stand = scaler.fit_transform(input_variables)

#Se decodifican las etiquetas de las categorias  
enc = OrdinalEncoder()
enc.fit(ouput_variable)
Var_categoricas_enc = enc.transform(ouput_variable)


#----------------------- Descoposición en componentes principales  

pca = PCA()
pca.fit(Var_numericas_stand)
componentes = pca.components_

datos_pca = pd.DataFrame(data    = pca.components_, columns = list_names)
#Mapa de Calor 
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
componentes = pca.components_
plt.imshow(componentes.T, cmap='magma', aspect='auto')
plt.yticks(range(len(list_names)), list_names)
plt.xticks(range(len(list_names)), np.arange(pca.n_components_) + 1)
plt.grid(False)
plt.colorbar();
#plt.show()

pca_array = np.array(componentes)

for row in range(len(pca_array)):
    comp = list(pca_array[row,:])
    maximo = max(comp)
    indice = comp.index(maximo)
    #print(list_names[indice])
    
#Varianza explicada por cada componente 
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(x = np.arange(pca.n_components_) + 1, height = pca.explained_variance_ratio_)

for x, y in zip(np.arange(len(list_names)) + 1, pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center')

ax.set_xticks(np.arange(pca.n_components_-30) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Porcentaje de varianza explicada por cada componente')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza explicada');
plt.show()


proyecciones = pca.transform(Var_numericas_stand)

#Proyeccion en dos dimensiones 
fig = plt.figure(figsize = (10, 7))
plt.scatter(proyecciones[:,1], proyecciones[:,2])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


#Proyección en 3 dimensiones 
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
p = ax.scatter3D(proyecciones[:,1], proyecciones[:,2], proyecciones[:,3])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()


proyecciones_frame = pd.DataFrame(proyecciones, columns = list_names)

proyecciones_frame.to_csv('PCA.csv')


