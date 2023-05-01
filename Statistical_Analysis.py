from numpy.core.fromnumeric import partition
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from scipy.stats import pearsonr, spearmanr 
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt 


# Se leen los datos y se clasifican entre variable de entrada y varaibles de salida
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


#Coeficiente de Correlación variables numericas 
print(len(list_names))
color = {"boxes": "Red",
"whiskers": "Orange",
"medians": "Blue",
"caps": "Gray"}
for i in range(len(list_names)):
    #print(list_names[i])
    r,p  = pearsonr(Var_numericas_stand[:,i],Var_categoricas_enc[:,0])
    #print(f"Correlación Pearson: r={r}, p-value={p}")
    r,p = spearmanr(Var_numericas_stand[:,i],Var_categoricas_enc[:,0])
    #print(f"Correlación Spearman: r={r}, p-value={p}")
    #input_variables[list_names[i]].plot.box(color = color)
    #plt.show()
    #print(input_variables[list_names[i]].describe())


#Prueba de Kolmogorv Smirnov 
distribucion = [ss.norm, ss.maxwell, ss.pareto, ss.t, ss.uniform, ss.beta,
               ss.logistic, ss.laplace]
distribucion_names =['norm', 'maxwell', 'pareto', 't', 'uniform', 'beta',
               'logistic', 'laplace']

pdf_names = [ss.norm.pdf, ss.maxwell.pdf, ss.pareto.pdf, ss.maxwell.pdf, ss.uniform.pdf, ss.laplace.pdf,
              ss.logistic.pdf, ss.laplace.pdf]             

def entropia(datos):
    H = [x*np.log(x) for x in datos]
    return -sum(H)              

for nombre in list_names: 
    parametros  = []
    distr = []
    pval = []
    for dist in range(len(distribucion)):
        param = distribucion[dist].fit(input_variables[nombre])
        d, pvalor = ss.kstest(input_variables[nombre],
                              distribucion_names[dist], param)
        parametros.append(param)
        distr.append(dist)
        pval.append(pvalor)
    minimo = max(pval)
    indice = pval.index(minimo)
    #print(nombre  + ': ' + distribucion_names[indice])
    data = ss.laplace.pdf(input_variables[nombre])
    #data  = pdf_names[indice](input_variables[nombre])
    #entropia_val = entropia(data)
    #print('Entropia: ' + str(entropia_val))
   

#Analisis de la varaible de salida 

pd.DataFrame(ouput_variable).value_counts().plot(kind='bar')
plt.show()

   