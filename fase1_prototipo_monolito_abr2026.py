# -*- coding: utf-8 -*-
"""
@author: daniel.martinez
"""

# =============================================================================
# LIBRERÍAS
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


# Semillas para reproducibilidad (tanto numpy como tensorflow)
np.random.seed(33)

# =============================================================================
# LOAD DATA
# =============================================================================

# CAMBIO: dejo solo el dataset que realmente estás usando y comento el resto como histórico.
# Si quieres cambiar ciudad / corte, solo cambias esta sección.

# Import dataset BASE DE PRUEBA 1
ruta = r'C:/Users/daniel.martinez/OneDrive - TIERRA Y ARMONIA CONSTRUCCION SA DE CV/Documentos/Escritorio/IDI 4/DIMES Softec/2022/GDL/'
data_ori = pd.read_excel(ruta + 'Guadalajara 4Q22.xlsx', header=0)
del ruta

#IDEA ORIGINAL: ingresar RP (Clasificación), DEPTO (tipo), MUNICIPIO, SQM, BHK, BATHS, LEVELS. 
#DUDAS:
    #Outliers cómo identificarlos? porque si me voy sólo a municipio voy a encontrar de múltiples niveles,
    #y si me voy a colonia voy a encontrar muy pocos subconjuntos (ver boxplot de colonia).

    ###Que porcentaje de info pierdo con colonia
    ###Si la mayoria en RP y menor a 10%
    
    #Estrategia outliers -> sqm/bedrooms >= 100. (proponer un valor)
    #Usar price/sqm para detección de outliers por location (colony)
    
    #En aquellas var desbalanceadas correcto proponer una transformación para normal. No aplica en la variable de salida

    #Correr modelos y medir accuracy quitando y combinando variables. 
    #Para esto por ej. mezclar Sold/Inventario (últimas uds más baras)
   
    #Para el escalamiento de X, también escalo las dummies? ya tuve que haber quitado sesgo de cuali y cuantitativas?

#%%
# =============================================================================
# ACOMODO Y REORGANIZACION BASE DE DATOS ORIGEN
# =============================================================================

df = data_ori.copy()

df_1 = df.iloc[:,0:8]
df_2 = df.iloc[:,44:57]
dfA_p = df.iloc[:,8:20]
dfB_p = df.iloc[:,20:32]
dfC_p = df.iloc[:,32:44]

dfA = pd.concat([df_1, dfA_p, df_2], axis=1) 
print("I shape dfA", dfA.shape)
dfA = dfA.rename(columns = {'Área A':'sqm','Terraza A':'terrace','Terreno A':'terrain',
                      'Recámara A':'bhk', 'Baños A':'baths','Alcoba A':'alcoba',
                      'Cuarto de Servicio A': 'room_serv', 'Cajón A':'park_u',
                      'Niveles A':'levels','Precio Actual A':'price',
                      'Precio Inicial A':'first_price','ValorM2 A':'price_per_sqm'}).dropna()
print("F shape dfA", dfA.shape)


dfB = pd.concat([df_1, dfB_p, df_2], axis=1) 
print("I shape dfB", dfB.shape)
dfB = dfB.rename(columns = {'Área B':'sqm','Terraza B':'terrace','Terreno B':'terrain',
                      'Recámara B':'bhk', 'Baños B':'baths','Alcoba B':'alcoba',
                      'Cuarto de Servicio B': 'room_serv', 'Cajón B':'park_u',
                      'Niveles B':'levels','Precio Actual B':'price',
                      'Precio Inicial B':'first_price','ValorM2 B':'price_per_sqm'}).dropna()
print("F shape dfB", dfB.shape)


dfC = pd.concat([df_1, dfC_p, df_2], axis=1) 
print("I shape dfC", dfC.shape)
dfC = dfC.rename(columns = {'Área C':'sqm','Terraza C':'terrace','Terreno C':'terrain',
                      'Recámara C':'bhk', 'Baños C':'baths','Alcoba C':'alcoba',
                      'Cuarto de Servicio C': 'room_serv', 'Cajón C':'park_u',
                      'Niveles C':'levels','Precio Actual C':'price',
                      'Precio Inicial C':'first_price','ValorM2 C':'price_per_sqm'}).dropna()
print("F shape dfC", dfC.shape)

data = pd.concat([dfA, dfB, dfC]).sort_values('ID',ascending=True)

data = data.rename(columns = {'ID':'id','Clasificación':'classification','Tipo':'type','Nombre':'project_name',
                      'Dirección':'address', 'Colonia':'colony','Municipio':'town',
                      'Promotor': 'promoter', 'Absorción':'absortion',
                      'Meses en venta':'months_in_sale','Meses de inventario':'inventory_months',
                      'Éxito Comercial':'comm_succ','Unidades Totales':'total_units',
                      'Unidades proyectadas':'master_plan_units','Inventario':'inventory',
                      'Vendidas':'sold_units','Fecha de Alta':'entry_date',
                      'Fecha de Inicio':'initial_date','Fecha de Actualización':'update_date',
                      'Fecha de Entrega':'delivery_date','Financiamiento 1a opción':'fin_op'}).dropna()

del df,df_1,df_2,dfA, dfB, dfC,dfA_p, dfB_p, dfC_p

# =============================================================================
# DATA CLEANING - COLONY & TYPE
# =============================================================================

# --- COLONY ---
data['colony'] = data['colony'].str.lower()

colony_patterns = [
    (data['colony'].str.contains('royal co', case=False, regex=False), 'royal country'),
    (data['colony'].str.contains('videncia', case=False, regex=False), 'providencia'),
    (data['colony'].str.contains('ladrón de guevar', case=False, regex=False), 'ladrón de guevara'),
    (data['colony'].str.contains('Villa Bosque (Villa Panamericana)', case=False, regex=False), 'villa bosque'),
    (data['colony'].str.contains('americana', case=False, regex=False), 'americana'),
    (data['colony'].str.contains('lafayette', case=False, regex=False), 'americana'),
    (data['colony'].str.contains('juan ocot', case=False, regex=False), 'san juan de ocotán'),
    (data['colony'].str.contains('santa ana tepet', case=False, regex=False), 'santa ana tepetitlán')
]

colony_criteria, colony_values = zip(*colony_patterns)
data['colony_group_1'] = np.select(colony_criteria, colony_values, None)
data['colony_group_1'] = data['colony_group_1'].combine_first(data['colony'])

data = data.drop(['colony'], axis='columns')
data = data.rename(columns={'colony_group_1': 'colony'})

# --- TYPE ---
data['type'] = data['type'].str.lower()

type_patterns = [
    (data['type'].str.contains('depto', case=False, regex=False), 'depto'),
    (data['type'].str.contains('loft', case=False, regex=False), 'depto'),
    (data['type'].str.contains('cd', case=False, regex=False), 'casa'),
    (data['type'].str.contains('cs', case=False, regex=False), 'casa'),
    (data['type'].str.contains('ch', case=False, regex=False), 'casa'),
    (data['type'].str.contains('town house', case=False, regex=False), 'casa')
]

type_criteria, type_values = zip(*type_patterns)
data['type_group_1'] = np.select(type_criteria, type_values, None)
data['type_group_1'] = data['type_group_1'].combine_first(data['type'])

data = data.drop(['type'], axis='columns')
data = data.rename(columns={'type_group_1': 'type'})

del colony_criteria, colony_patterns, colony_values, type_criteria, type_patterns, type_values


# =============================================================================
# DATA QUALITY REPORT
# =============================================================================

def dqr(df):
    """Simple Data Quality Report para el dataframe."""
    cols = pd.DataFrame(list(df.columns.values),
                        columns=['Names'],
                        index=list(df.columns.values))
    dtyp = pd.DataFrame(df.dtypes, columns=['Type'])
    misval = pd.DataFrame(df.isnull().sum(), columns=['Missing_values'])
    presval = pd.DataFrame(df.count(), columns=['Present_values'])
    unival = pd.DataFrame(columns=['Unique_values'])
    minval = pd.DataFrame(columns=['Min_value'])
    maxval = pd.DataFrame(columns=['Max_value'])

    for col in df.columns:
        unival.loc[col] = [df[col].nunique()]
        try:
            minval.loc[col] = [df[col].min()]
            maxval.loc[col] = [df[col].max()]
        except Exception:
            pass

    return cols.join(dtyp).join(misval).join(presval).join(unival).join(minval).join(maxval)


report_data = dqr(data)

#%%

# =============================================================================
# LIMPIEZA / SELECCIÓN DE VARIABLES (A NIVEL DATASET)
# =============================================================================

#VARIABLES ELIMINADAS:
#Descriptivas del proyecto: ID,NOMBRE,DIRECCION,PROMOTER (si soy nuevo en mercado no me importa saber quien es quien)
#De cada prototipo: ALCOBA, CUARTO DE SERVICIO, PRECIO INICIAL -> muy pocos datos, Ninguna unidad Casa o Depa con Room_Serv, importa solo el actual no su politica de incrementos
#Del desarrollo/desarrollador: 
    #FECHA DE INICIO -> da mismo dato que MESES EN VENTA (tiempo transcurrido desde inicio de venta)
    #FECHA ALTA -> no importa cuando lo capturó DIME en su radar
    #FECHA DE ACTUALIZACION -> asumo los precios son iniciales vigentes por el 3Q-2021 -> actualizados cuatrimestralmente (regla de negocio)
    #FINANCIAMIENTO 1A OPCION -> desarrolladores por comodidad de papeleos escogen banco pero no impacta precios

#NOTAS:
#Inventario (uds disponibles a la venta que pueden o no estar construidas) REPRESENTA CUANTO ME QUEDA DISPONIBLE Y MIENTRAS MENOR SEA PENSARIA MAYOR EL PRECIO
#Vendidas (uds ya vendidas a la fecha de actualización)
#Unidades proyectadas (Master Plan Units aunque puedan o no estar construidas y si o no en venta) Es el tamaño de desarrollo.
#Unidades totales (número de uds existentes en la etapa disponible (pueden o no estar construidas))

# NOTA IMPORTANTE:
# Dejo aquí solo las variables realmente "muertas" para todo el análisis.
# Las que aún usamos en EDA (como comm_succ, price_per_sqm)
# las eliminaremos DESPUÉS de las gráficas que las necesitan.

###CRITERIO DE NEGOCIO (CON CONOCIMIENTO DE NEGOCIO Y CONTEXTO DE PROBLEMATICA)
#Elimino variables que no tienen que ver con la variable salida
data = data.drop([
    'id', 'project_name', 'address', 'promoter', 'alcoba', 'room_serv', 'first_price',
    'initial_date', 'entry_date', 'update_date', 'fin_op',
    'inventory_months'
], axis='columns', errors='ignore')
print("Columnas tras primera limpieza:", data.columns.tolist())

report_data = dqr(data)

#CALAR
#Unidades totales -> Tamaño del phase plan
#Meses de inventario = Inventario/Absorción (tiempo estimado para al ritmo de absorción acabarse Inventario)
#Exito comercial = Absorción/Unidades Totales

#OJO:
    #ABSORCION ES DE LOS 3 O 2 O 1 PROTOTIPOS
    #MESES DE INVENTARIO ES DE LOS 3,2 O 1 PROTOTIPOS
    #UNIDADES PROYECTADAS #No hay tema pues es del proyecto
    #UNIDADES TOTALES #No hay tema pues es del proyecto
    #INVENTARIO
    #VENDIDAS

# =============================================================================
# FEATURE ENGINEERING: MESES PARA ENTREGA
# =============================================================================

from dateutil.relativedelta import relativedelta  # CAMBIO: usamos relativedelta para meses exactos

# Fecha base de referencia
base_date = pd.to_datetime('22/09/01', format='%y/%m/%d')
data['delivery_date'] = pd.to_datetime(data['delivery_date'])

def months_diff(fecha):
    """
    Calcula la diferencia en meses (aprox) entre fecha de entrega y base_date.
    Incluye años*12 + meses + fracción de mes por días.
    """
    rd = relativedelta(fecha, base_date)
    return rd.years * 12 + rd.months + rd.days / 30.44   # CAMBIO: evitamos usar np.timedelta64('M')

# Creamos la variable de meses a entrega
data['months_to_delivery'] = data['delivery_date'].apply(months_diff)

# Si el proyecto ya debería estar entregado, truncamos a 0 meses
data.loc[data['months_to_delivery'] < 0, "months_to_delivery"] = 0

# Ya no necesitamos la fecha de entrega como tal para el modelo
data = data.drop(['delivery_date'], axis='columns')

#CONVERTIR A % SOLD_UNITS (AVANCE DEL TOTAL MASTER_PLAN)"""
#Esta variable espero esté muy correlacionada con meses en venta y sólo quedarme con una.

#%%


# =============================================================================
# FEATURE ENGINEERING: CLASIFICACIÓN SOFTEC A NUMÉRICO
# =============================================================================

#"Clase 5 Ing Características"
#Clasificación SOFTEC (S-Social, E-Economica, M-Media, R-Residencial, RP-Residencial Plus)
clas_soft = ['S','E','M','R','RP']

#L = len(data['Clasificación'].unique()) #Muestra las diferentes categorías en la columna
L = len(clas_soft) #Muestra las diferentes categorías en la columna
div=L/2 if L%2==0 else (L-1)/2

#Para hacer un mapeo entre categorías y números se usa un diccionario
#mapeo={k:i for i,k in enumerate(data['Clasificación'].unique(),int(-div))}
#mapeo={k:2*i+1 for i,k in enumerate(clas_soft,int(-div))}
mapeo={k:i for i,k in enumerate(clas_soft,int(-div))}
print(mapeo)

data['classification'] = data['classification'].map(mapeo)

del L, mapeo, div, clas_soft

#%%
# =============================================================================
# FILTRO A DEPTOS Y CIUDADES CON POCA MUESTRA
# =============================================================================

#Como lo que nos interesa es el modelo aplicable para producto departamentos, nos quedamos con esa data
data = data[data['type'] == 'depto']
data = data.drop(['terrain', 'type'], axis='columns')  # Terreno correlacionado con sqm

# Elimino ciudades con muy pocos datos (menos de 10 productos distintos)
data = data.drop(data[data['town'] == 'Chapala'].index)
data = data.drop(data[data['town'] == 'El Salto'].index)

#%%
# =============================================================================
# EDA – HISTOGRAMAS, SESGO, KURTOSIS
# =============================================================================

##SESGO DA IGUAL YA SEA CON O SIN ESCALAMIENTO

# Me quedo solo con columnas numéricas para este análisis
data_num = data.select_dtypes(include=['number'])  # CAMBIO: evita strings como 'Guadalajara'

# Histogramas
data_num.hist(bins=20)
plt.tight_layout()

# Skew y kurtosis solo en numéricas
data_skew_i = pd.DataFrame([data_num.skew(numeric_only=True)]).transpose()
data_skew_i = data_skew_i.rename(columns={0: 'Skew'})

data_kurt_i = pd.DataFrame([data_num.kurtosis(numeric_only=True)]).transpose()
data_kurt_i = data_kurt_i.rename(columns={0: 'Kurtosis'})

tabla_skew_kurt_i = (
    pd.concat([data_skew_i, data_kurt_i], axis=1)
      .sort_values('Skew', ascending=False)
)

del data_skew_i, data_kurt_i

sb.set(style="darkgrid")

# Si quiero seguir usando data_hist_i, hacerlo a partir de numéricas
data_hist_i = data_num.copy()   # CAMBIO: ya no incluyo 'town', 'colony'

# Ejemplo: histograma de la columna 7 (ajusta si cambia el orden)
sb.histplot(data_hist_i.iloc[:, 7], kde=True, color="olive")
plt.show()

# =============================================================================
# BOXPLOTS Y SCATTERS
# =============================================================================

data.boxplot(column=['price'], by='classification', figsize=(25, 6), rot=96)
data.boxplot(column=['price'], by='town', figsize=(25, 6), rot=96)
data.boxplot(column=['price'], by='bhk', figsize=(25, 6), rot=96)
data.boxplot(column=['price'], by='colony', figsize=(25, 6), rot=96)

data.plot.scatter(x='classification', y='price', c='blue', figsize=(20, 6))
data.plot.scatter(x='bhk', y='price', c='blue', figsize=(20, 6))

areatype_stats = data.groupby('classification')['classification'].agg('count').sort_values(ascending=False)
location_stats = data.groupby('town')['town'].agg('count').sort_values(ascending=False)
colony_stats = data.groupby('colony')['colony'].agg('count').sort_values(ascending=False)
bhk_stats = data.groupby('bhk')['bhk'].agg('count').sort_values(ascending=False)


#%%

# =============================================================================
# VISUALIZACIÓN: SUPERFICIE VS PRECIO POR BHK
# =============================================================================

#VISUALIZACION POR CUARTOS DE SUPERFICIE VS PRECIO TOTAL
def plot_total_sqm_vs_price_chart(df, town):
    bhk1 = df[(df.town == town) & (df.bhk == 1)]
    bhk2 = df[(df.town == town) & (df.bhk == 2)]
    bhk3 = df[(df.town == town) & (df.bhk == 3)]
    bhk4 = df[(df.town == town) & (df.bhk == 4)]
    plt.scatter(bhk1.sqm, bhk1.price, color='red', label='1 BHK', s=10)
    plt.scatter(bhk2.sqm, bhk2.price, color='blue', label='2 BHK', s=10)
    plt.scatter(bhk3.sqm, bhk3.price, marker='+', color='green', label='3 BHK', s=10)
    plt.scatter(bhk4.sqm, bhk4.price, color='black', label='4 BHK', s=10)
    plt.xlabel("Total Sqm Area")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


plot_total_sqm_vs_price_chart(data, "Guadalajara")


# =============================================================================
# VISUALIZACIÓN: SEGMENTO VS PRECIO M2
# =============================================================================

#VISUALIZACION POR CUARTOS DE SUPERFICIE VS PRECIO UNITARIO
def plot_type_vs_price_per_sqm(df, town):
    RP = df[(df.town == town) & (df.classification == 2)]
    R = df[(df.town == town) & (df.classification == 1)]
    M = df[(df.town == town) & (df.classification == 0)]
    E = df[(df.town == town) & (df.classification == -1)]
    S = df[(df.town == town) & (df.classification == -2)]
    plt.scatter(RP.sqm, RP.price_per_sqm, color='blue', label='RP', s=25)
    plt.scatter(R.sqm, R.price_per_sqm, marker='+', color='green', label='R', s=25)
    plt.scatter(M.sqm, M.price_per_sqm, marker='*', color='red', label='M', s=25)
    plt.scatter(E.sqm, E.price_per_sqm, marker='+', color='purple', label='E', s=25)
    plt.scatter(S.sqm, S.price_per_sqm, marker='*', color='orange', label='S', s=25)
    plt.xlabel("Total sqm Area")
    plt.ylabel("Price per sqm")
    plt.legend()
    plt.show()


plot_type_vs_price_per_sqm(data, "Tlaquepaque")


# =============================================================================
# VISUALIZACIÓN: ÉXITO COMERCIAL VS PRECIO M2
# =============================================================================

def plot_total_sqft_vs_pps(df, town):
    RP = df[(df.town == town) & (df.classification == 2)]
    R = df[(df.town == town) & (df.classification == 1)]
    M = df[(df.town == town) & (df.classification == 0)]
    E = df[(df.town == town) & (df.classification == -1)]
    S = df[(df.town == town) & (df.classification == -2)]
    plt.scatter(RP.comm_succ, RP.price_per_sqm, color='blue', label='RP', s=25)
    plt.scatter(R.comm_succ, R.price_per_sqm, marker='+', color='green', label='R', s=25)
    plt.scatter(M.comm_succ, M.price_per_sqm, marker='*', color='red', label='M', s=25)
    plt.scatter(E.comm_succ, E.price_per_sqm, marker='+', color='purple', label='E', s=25)
    plt.scatter(S.comm_succ, S.price_per_sqm, marker='*', color='orange', label='S', s=25)
    plt.xlabel("Commercial Success (Abs/Tot_units)")
    plt.ylabel("Price per sqm")
    plt.legend()
    plt.show()


plot_total_sqft_vs_pps(data, "Zapopan")


#%%

# =============================================================================
# PROCESO EXPLORACIÓN / PCA / CORRELACIONES
# =============================================================================

from sklearn import preprocessing

subdata_X = data.copy()
#Quito variables cualitativas (DUMMIES: TYPE, COLONY/TOWN  CONVERSION NUMERICA -> CLASSFICATION)
subdata_X = subdata_X.drop(['colony', 'town', 'price'], axis='columns')
subdata_X_scaled = (subdata_X - subdata_X.mean()) / subdata_X.std()

#Variables que tienen más varianza para el conjunto de datos

# Dado que no quiero graficar price_per_sqm, hago un pequeño tratamiento donde excluyo porque domina por escala y donde me quedo solo con variables numéricas
subdata_X_num = subdata_X.select_dtypes(include=['number']).copy()
cols_var = [c for c in subdata_X_num.columns if c != 'price_per_sqm']
variances = subdata_X_num[cols_var].var().sort_values()
plt.figure(figsize=(10, 8))
plt.bar(np.arange(len(variances)), variances.values)
plt.ylabel('Variance')
plt.xticks(np.arange(len(variances)), variances.index, rotation=50)
plt.tight_layout()
plt.show()


# Correlation entre variables independientes con la variable dependiente
subdata_Xy = data.copy()
subdata_Xy = subdata_Xy.drop(['classification', 'colony', 'town'], axis='columns')
subdata_Xy.corr()[['price']].sort_values(by='price', ascending=False)

plt.figure(figsize=(8, 12))
heatmap = sb.heatmap(subdata_Xy.corr()[['price']].sort_values(by='price', ascending=False),
                     vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Price', fontdict={'fontsize': 18}, pad=16)

#%%
# =============================================================================
# PCA Origen
# =============================================================================

#Objetivo: encontrar qué variables de mi conjunto original son las más importantes
#TECNICA DE REDUCCION DIMENSIONAL

from sklearn.decomposition import PCA
#from sklearn import preprocessing
#tmp = subdata_X
tmp = preprocessing.scale(subdata_X)
#pca = PCA(n_components=10)
pca = PCA(0.95)
#pca = PCA()
pca.fit(tmp)
data_pca = pca.transform(tmp)
componentes = pca.components_
# print(subdata_X.columns)
# print(componentes)

pca_per = pca.explained_variance_ratio_

pca_per = ["%.3f" % x for x in pca_per]
print(pca_per)
print(subdata_X.columns)
print('Se revisa la matriz componentes y por c/componente principal se estudia la variable más injerente')


#CERIORARME ESTEN LAS VARIABLES EN EL ORDEN DE SUBDATA
# data_pca = pd.DataFrame(data_pca, columns = ['sqm*', 'terrace*', 'terrain*', 'bhk*', 'baths*', 'park_u*', 'levels*',
#        'absortion*', 'months_in_sale*', 'comm_succ*', 'proj_units*', 'total_units*','inventory*', 'sold_units*'])
# fig = sb.pairplot(data_pca)

#ELIMINAR SOLD_UNITS Y INVENTORY. 

#%%
# =============================================================================
# ANALISIS DE CORRELACION: CLUSTERING DE VARIABLES + MAPA DE CALOR ORDENADO
# =============================================================================

#The pearson correlation coefficient values are estimated between each features 
#including the target feature. These values range between -1 and 1. While 
#values near -1 implies that two variables are inversely proportional, values 
#near 1 implies that two variables are directly proportional. On the other hand, 
#the values near 0 indicates that there is not significant correlation between variables. 


# Correlaciones numéricas
plt.figure(figsize=(16, 6))
heatmap = sb.heatmap(subdata_X.corr(method='spearman'), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap I', fontdict={'fontsize': 12}, pad=12)


from scipy.cluster.hierarchy import dendrogram, linkage

# Asegurarnos que solo tenemos variables numéricas
subdata_X_num = subdata_X.select_dtypes(include=['number'])

# ---------- DENDROGRAMA ----------
Z = linkage(subdata_X_num.T, metric='correlation', method='complete')

plt.figure(figsize=(10, 4))  # Figura más horizontal
dendro = dendrogram(
    Z,
    labels=subdata_X_num.columns,  # mostramos nombres de variables
    leaf_rotation=90               # giramos etiquetas para que se lean
)
plt.title("Dendrograma de variables (métrica: correlación)")
plt.tight_layout()
plt.show()

# ---------- REORDENAR MATRIZ DE CORRELACIÓN SEGÚN EL DENDROGRAMA ----------
# dendro['leaves'] trae los índices reordenados
ordered_cols = [subdata_X_num.columns[i] for i in dendro['leaves']]

# Recalculamos la matriz de correlación en ese orden
corr_reordered = subdata_X_num[ordered_cols].corr(method='spearman')

# ---------- HEATMAP ORDENADO ----------
plt.figure(figsize=(8, 6))
sb.heatmap(
    corr_reordered,
    vmin=-1, vmax=1,
    cmap='coolwarm',
    square=True,                 # cuadritos cuadrados
    xticklabels=ordered_cols,
    yticklabels=ordered_cols
)
plt.title("Mapa de calor de correlación (ordenado por clustering)", fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# TRATAMIENTO OUTLIERS (NEGOCIO)
# =============================================================================

#Criterio A: seleccionado en base a no pérdida masiva de datos y perspectiva de negocio
data = data[(data.price_per_sqm < 90000) & (data.sqm < 250.0)]

#Criterio B o futuro: HACER SEGMENTADOR DEL PREDICTOR FUERA POR COLONIA (MUCHOS MAS DATOS)
# def remove_pps_outliers(df):
#     df_out = pd.DataFrame()
#     for key, subdf in df.groupby('colony'):
#         Q1=subdf.price.quantile(0.25)
#         Q3=subdf.price.quantile(0.75)
#         Ri=Q3-Q1        
#         reduced_df = subdf[(subdf.price>(Q1-1.5*Ri)) & (subdf.price<=(Q3+1.5*Ri))]
#         df_out = pd.concat([df_out,reduced_df], ignore_index=True)
#     return df_out

# data = remove_pps_outliers(data)


#areatype_stats = data.groupby('classification')['classification'].agg('count').sort_values(ascending=False)
# def remove_outliers_ppsqm(df):
#     df_out = pd.DataFrame()
#     for key, subdf in df.groupby('town'):
#         Q1=subdf.price_per_sqm.quantile(0.25)
#         Q3=subdf.price_per_sqm.quantile(0.75)
#         Ri=Q3-Q1        
#         reduced_df = subdf[(subdf.price_per_sqm>(Q1-1.5*Ri)) & (subdf.price_per_sqm<=(Q3+1.5*Ri))]
#         df_out = pd.concat([df_out,reduced_df], ignore_index=True)
#     return df_out

# data = remove_outliers_ppsqm(data)

#areatype_stats_2 = data2.groupby('classification')['classification'].agg('count').sort_values(ascending=False)

# #BAJO ESTE CRITERIO PIERDO CERCA DEL 25% DE RP (27 DATOS) 
# areatype_stats_3 = data.groupby('classification')['classification'].agg('count').sort_values(ascending=False)

# def remove_bhk_outliers(df):
#     exclude_indices = np.array([])
#     for town ,town_df in df.groupby('town'):
#         classification_stats= {}
#         for classification, classification_df in town_df.groupby('classification'):
#             classification_stats[classification] = {
#                     'mean': np.mean(classification_df.price_per_sqm),
#                     'std': np.std(classification_df.price_per_sqm),
#                     'count': classification_df.shape[0]
#             }
#         for classification, classification_df in town_df.groupby('classification'):
#             stats = classification_stats.get(classification-1)
#             if stats and stats['count']>5:
#                 exclude_indices = np.append(exclude_indices, classification_df[classification_df.price_per_sqm<(stats['mean'])].index.values)
#     return df.drop(exclude_indices, axis='index')

# data_w_o = remove_bhk_outliers(data)   

# areatype_stats_4 = data_w_o.groupby('classification')['classification'].agg('count').sort_values(ascending=False)



#%%
# =============================================================================
# TRATAMIENTO SKEW (BOX-COX, SOLO EXPLORATORIO)
# =============================================================================

from scipy import stats

# Copia para experimentar con Box-Cox sin modificar "data" original
data_copy = data.copy()

# Aplicamos Box-Cox a columnas numéricas positivas
# OJO: Box-Cox requiere valores > 0, si hay ceros o negativos, se desplazan
for col in ['sqm', 'months_in_sale', 'master_plan_units']:
    col_min = data_copy[col].min()
    if col_min <= 0:
        # CAMBIO: desplazamos la serie para que todos los valores sean > 0
        data_copy[col] = data_copy[col] - col_min + 1e-3
    data_copy[col], lamb = stats.boxcox(data_copy[col])
    print(f"Lambda Box-Cox para {col}: {lamb}")

# ---- EDA DESPUÉS DEL TRATAMIENTO (SOLO NUMÉRICAS) ----

# Usamos únicamente columnas numéricas de data_copy
data_num_f = data_copy.select_dtypes(include=['number'])

data_num_f.hist(bins=20)
plt.tight_layout()

data_skew_f = (
    pd.DataFrame([data_num_f.skew(numeric_only=True)])
    .transpose()
    .rename(columns={0: 'Skew'})
)

data_kurt_f = (
    pd.DataFrame([data_num_f.kurtosis(numeric_only=True)])
    .transpose()
    .rename(columns={0: 'Kurtosis'})
)

tabla_skew_kurt_f = (
    pd.concat([data_skew_f, data_kurt_f], axis=1)
      .sort_values('Skew', ascending=False)
)

del data_skew_f, data_kurt_f


# =============================================================================
# DUMMIES TOWN Y PREPARACIÓN PARA MODELADO
# =============================================================================

dummies = pd.get_dummies(data.town)
data = pd.concat([data, dummies], axis='columns')

# CAMBIO IMPORTANTE:
# A partir de aquí ya no necesitamos price_per_sqm ni comm_succ para el modelo,
# pero sí estuvieron disponibles para la EDA anterior.
data = data.drop(['colony', 'town', 'price_per_sqm', 'comm_succ', 'absortion', 'sold_units', 'baths'],
                 axis='columns', errors='ignore')


#VARIABLES ELIMINADAS:
#SOLD UNITS: Correlacionada 0.96 con Total Units ("normal" que desarrollos viejos han vendido más)-> En este caso meses en venta da más valor
#price_per_sqm sirve para detectar outliers pero es dependiente de price y no sería un dato conocido al usar modelo
#BATHS: Muy correlacionada a SQM (el número de baños no aporta algo que los m2 construidos no lo hagan)
#ABSORTION: No hay una componente principal (de las mayores) que tenga mucho peso y además está muy correlacionada con 

#%%
# =============================================================================
# SELECCIÓN X E Y Y ESCALADO SIN DATA LEAKAGE (CON FEATURE NAMES)
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataM = data.copy()

# 1) Mantener un DataFrame para conservar nombres de variables
X_df = dataM.drop(['price'], axis='columns')

# 2) Guardar los nombres de las features para análisis posterior (ej. coeficientes Ridge)
feature_names = X_df.columns.tolist()

# 3) Convertir a numpy para los modelos
X = X_df.values
Y = dataM['price'].values.reshape(-1, 1)

# 4) Split train/test
xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 5) Escalado sin data leakage
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

xtrain = scaler_X.fit_transform(xtrain)
xtest = scaler_X.transform(xtest)

ytrain = scaler_Y.fit_transform(ytrain)
ytest = scaler_Y.transform(ytest)

#%%

# =============================================================================
# MODELO 1: LINEAR REGRESSION
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lr_model = LinearRegression()
lr_model.fit(xtrain, ytrain)

r_sq_train = lr_model.score(xtrain, ytrain)
r_sq_test = lr_model.score(xtest, ytest)

print("Coeficiente de determinación train:", r_sq_train)
print("Coeficiente de determinación test:", r_sq_test)
print("Intercepto:", lr_model.intercept_)
print("Slope:", lr_model.coef_)

scores = cross_val_score(lr_model, xtrain, ytrain, scoring='r2', cv=20)
print("Cross_val_score")
print(scores)

# R2 train
Yhat = lr_model.predict(xtrain)
R2_score_tr = r2_score(ytrain, Yhat)

xmin, xmax = np.min(ytrain), np.max(ytrain)
xline = np.linspace(xmin, xmax)
plt.figure(figsize=(10, 6))
plt.scatter(ytrain, Yhat, label='Estimation')
plt.plot(xline, xline, 'k--', label='Perfect estimation')
plt.xlabel('Real output', fontsize=20)
plt.ylabel('Estimation output', fontsize=20)
plt.title('R^2 train=%0.4f' % R2_score_tr, fontsize=20)
plt.legend()
plt.grid()
plt.show()

# R2 test
Yhat_t = lr_model.predict(xtest)
R2_score_ts = r2_score(ytest, Yhat_t)

xmin, xmax = np.min(ytest), np.max(ytest)
xline = np.linspace(xmin, xmax)
plt.figure(figsize=(10, 6))
plt.scatter(ytest, Yhat_t, label='Estimation')
plt.plot(xline, xline, 'k--', label='Perfect estimation')
plt.xlabel('Real output', fontsize=20)
plt.ylabel('Estimation output', fontsize=20)
plt.title('R^2 test=%0.4f' % R2_score_ts, fontsize=20)
plt.legend()
plt.grid()
plt.show()


# =============================================================================
# RFE + GRIDSEARCH SOBRE LR (PARA TUNEO DE FEATURES)
# =============================================================================

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import RFE

folds = KFold(n_splits=10, shuffle=True, random_state=42)  # CAMBIO: añado random_state
hyper_params = [{'n_features_to_select': list(range(1, 16))}]

lm = LinearRegression()
rfe = RFE(lm)

model_cv = GridSearchCV(estimator=rfe,
                        param_grid=hyper_params,
                        scoring='r2',
                        cv=folds,
                        verbose=1,
                        return_train_score=True)

model_cv.fit(xtrain, ytrain)

cv_results = pd.DataFrame(model_cv.cv_results_)

plt.figure(figsize=(16, 6))
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')

#%%
# =============================================================================
# MODELO 2: ÁRBOL SIMPLE
# =============================================================================

from sklearn.tree import DecisionTreeRegressor

model2 = DecisionTreeRegressor(random_state=0,
                               splitter='best',
                               max_depth=None,
                               min_samples_split=2,
                               min_samples_leaf=1)
X_dt = xtrain.copy()
Y_dt = ytrain.copy()

model2 = model2.fit(X_dt, Y_dt)
Yhat = model2.predict(X_dt)
print('R2 (train árbol) = %0.4f' % model2.score(X_dt, Y_dt))

plt.figure(figsize=(8, 8))
plt.scatter(Y_dt, Yhat, c='r', label='datos-ruido')
plt.plot(np.linspace(np.min(Y_dt), np.max(Y_dt), 100),
         np.linspace(np.min(Y_dt), np.max(Y_dt), 100),
         'g--', label='ajuste')
plt.xlabel('Y_real')
plt.ylabel('Y_estimada')
plt.grid()
plt.legend()
plt.show()

# Búsqueda sobre profundidad
depths = np.arange(15)
models = []
model_scores = np.zeros(np.shape(depths))
for md in depths:
    model_tmp = DecisionTreeRegressor(random_state=0,
                                      splitter='best',
                                      max_depth=md + 1,
                                      min_samples_split=2,
                                      min_samples_leaf=1)
    model_tmp = model_tmp.fit(X_dt, Y_dt)
    model_scores[md] = model_tmp.score(X_dt, Y_dt)
    models.append(model_tmp)

grad_score = np.diff(model_scores)

plt.figure()
plt.plot(depths + 1, model_scores)
plt.xlabel('Max_depths'), plt.ylabel('R^2 score')
plt.grid()

plt.figure()
plt.plot(depths[1:] + 1, grad_score)
plt.xlabel('Max_depths'), plt.ylabel('diff_score')
plt.grid()

A = model2.feature_importances_
Var_prin = np.argsort(A)
depth = model2.get_depth()

# Evaluación árbol en train/test
Yhat_tr = model2.predict(xtrain)
R2_score_tr_tree = r2_score(ytrain, Yhat_tr)

plt.figure(figsize=(10, 6))
plt.scatter(ytrain, Yhat_tr, label='Estimation')
plt.plot(np.linspace(np.min(ytrain), np.max(ytrain), 100),
         np.linspace(np.min(ytrain), np.max(ytrain), 100),
         'k--', label='Perfect estimation')
plt.xlabel('Real output', fontsize=20)
plt.ylabel('Estimation output', fontsize=20)
plt.title('R^2 árbol train=%0.4f' % R2_score_tr_tree, fontsize=20)
plt.legend()
plt.grid()
plt.show()

Yhat_ts = model2.predict(xtest)
R2_score_ts_tree = r2_score(ytest, Yhat_ts)

plt.figure(figsize=(10, 6))
plt.scatter(ytest, Yhat_ts, label='Estimation')
plt.plot(np.linspace(np.min(ytest), np.max(ytest), 100),
         np.linspace(np.min(ytest), np.max(ytest), 100),
         'k--', label='Perfect estimation')
plt.xlabel('Real output', fontsize=20)
plt.ylabel('Estimation output', fontsize=20)
plt.title('R^2 árbol test=%0.4f' % R2_score_ts_tree, fontsize=20)
plt.legend()
plt.grid()
plt.show()

#%%
# =============================================================================
# MODELO 3: BAGGING TREES
# =============================================================================

from sklearn.ensemble import BaggingRegressor
import time

# CAMBIO IMPORTANTE:
# Antes entrenaba Bagging con TODA la X,Y (overfitting muy fuerte).
# Ahora lo entreno con xtrain,ytrain y evaluamos en train/test.

ytrain_vec = ytrain.ravel()
ytest_vec = ytest.ravel()

bag_model = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=50,
    random_state=0,
    max_samples=0.7,
    oob_score=False,
    verbose=0
)

start_time = time.time()
bag_model = bag_model.fit(xtrain, ytrain_vec)
print("--- %s seconds ---" % (time.time() - start_time))

Yhat_bag_tr = bag_model.predict(xtrain)
print('R2 bagging train = %0.4f' % bag_model.score(xtrain, ytrain_vec))
R2_bag_tr = r2_score(ytrain_vec, Yhat_bag_tr)

plt.figure(figsize=(8, 8))
plt.scatter(ytrain_vec, Yhat_bag_tr, c='r', label='data-train')
plt.plot(np.linspace(np.min(ytrain_vec), np.max(ytrain_vec), 100),
         np.linspace(np.min(ytrain_vec), np.max(ytrain_vec), 100),
         'g--', label='fit')
plt.xlabel('y_real')
plt.ylabel('Y_estimated')
plt.title('R^2 train bagging=%0.4f' % R2_bag_tr, fontsize=20)
plt.legend()
plt.grid()
plt.show()

# GridSearch para Bagging
parameters = {
    'n_estimators': [40, 50, 60],
    'max_samples': [0.70, 0.75, 0.80],
}

bag_base = BaggingRegressor(random_state=42)

bt = GridSearchCV(bag_base, parameters)
bt.fit(xtrain, ytrain_vec)

print("Mejor estimador Bagging:", bt.best_estimator_)
print("R2 Bagging train (best):", bt.score(xtrain, ytrain_vec))

y_pred_train = bt.predict(xtrain)
R2_score_bag_tr = r2_score(ytrain_vec, y_pred_train)
print('Train data R^2 bagging tuned = %0.4f' % R2_score_bag_tr)

plt.figure(figsize=(8, 8))
plt.scatter(ytrain_vec, y_pred_train, c='r', label='data-train')
plt.plot(np.linspace(np.min(ytrain_vec), np.max(ytrain_vec), 100),
         np.linspace(np.min(ytrain_vec), np.max(ytrain_vec), 100),
         'g--', label='fit')
plt.xlabel('y_real')
plt.ylabel('Y_estimated')
plt.title('R^2 train bagging tuned=%0.4f' % R2_score_bag_tr, fontsize=20)
plt.legend()
plt.grid()
plt.show()

y_pred_test = bt.predict(xtest)
R2_score_bag_ts = r2_score(ytest_vec, y_pred_test)
print('Test data R^2 bagging tuned = %0.4f' % R2_score_bag_ts)

plt.figure(figsize=(8, 8))
plt.scatter(ytest_vec, y_pred_test, c='r', label='data-test')
plt.plot(np.linspace(np.min(ytest_vec), np.max(ytest_vec), 100),
         np.linspace(np.min(ytest_vec), np.max(ytest_vec), 100),
         'g--', label='fit')
plt.xlabel('y_real')
plt.ylabel('Y_estimated')
plt.title('R^2 test bagging tuned=%0.4f' % R2_score_bag_ts, fontsize=20)
plt.legend()
plt.grid()
plt.show()

# RMSE y bias para Bagging
rmse_train_bag = root_mean_squared_error(ytrain_vec, y_pred_train)
rmse_test_bag  = root_mean_squared_error(ytest_vec,  y_pred_test)
print('Train data RMSE BT= %0.4f' % rmse_train_bag)
print('Test data RMSE BT= %0.4f' % rmse_test_bag)

bias_train_bag = np.mean(y_pred_train - ytrain_vec)
bias_test_bag = np.mean(y_pred_test - ytest_vec)
print('Train data bias BT = %0.4f' % bias_train_bag)
print('Test data bias BT = %0.4f' % bias_test_bag)

#%%
# =============================================================================
# MODELO 4: NEURAL NETWORK TF.KERAS
# =============================================================================

#MODIFICAR TODO ESTO.

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error  # Asegúrate de tenerlo importado arriba

# Número de epochs (equivalente a los epochs de Keras)
epochs = 200

# sklearn trabaja más cómodo con vectores 1D
ytrain_nn = ytrain.ravel()
ytest_nn = ytest.ravel()

# Definimos una red similar a la que tenías en tf.keras:
# - 1 capa oculta con 6 neuronas
# - activación ReLU
# - solver Adam
# - learning_rate_init ≈ 0.005
# - max_iter=1 + warm_start=True para simular entrenamiento por epochs
mlp = MLPRegressor(
    hidden_layer_sizes=(6,),
    activation='relu',
    solver='adam',
    learning_rate_init=0.005,
    max_iter=1,          # una iteración por "epoch"
    warm_start=True,     # reusa pesos entre llamadas a fit
    random_state=33
)

train_mse_history = []
val_mse_history = []

for epoch in range(epochs):
    # Entrenamos 1 "epoch" más
    mlp.fit(xtrain, ytrain_nn)

    # Predicciones en train y test
    yhat_tr = mlp.predict(xtrain)
    yhat_ts = mlp.predict(xtest)

    # MSE en train y validación
    mse_train = mean_squared_error(ytrain_nn, yhat_tr)
    mse_val = mean_squared_error(ytest_nn, yhat_ts)

    train_mse_history.append(mse_train)
    val_mse_history.append(mse_val)

    # (Opcional) log cada cierto número de epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} - MSE train: {mse_train:.4f} - MSE val: {mse_val:.4f}")

# -------------------------------------------------------------------------
# Métricas finales de la red neuronal
# -------------------------------------------------------------------------
R2_nn_train = r2_score(ytrain_nn, yhat_tr)
R2_nn_test  = r2_score(ytest_nn, yhat_ts)

rmse_nn_train = root_mean_squared_error(ytrain_nn, yhat_tr)
rmse_nn_test  = root_mean_squared_error(ytest_nn, yhat_ts)

print("\n===== Neural Network (MLPRegressor) =====")
print("Train R² NN = %.4f" % R2_nn_train)
print("Test  R² NN = %.4f" % R2_nn_test)
print("Train RMSE NN = %.4f" % rmse_nn_train)
print("Test  RMSE NN = %.4f" % rmse_nn_test)

# -------------------------------------------------------------------------
# Gráficas: MSE y RMSE por epoch (equivalentes a history de Keras)
# -------------------------------------------------------------------------
# MSE
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(train_mse_history, 'r', label='train MSE')
ax.plot(val_mse_history, 'b', label='val MSE')
ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
ax.legend()
ax.set_title('Evolución MSE por epoch (NN - MLPRegressor)', fontsize=16)
ax.grid(True)
plt.tight_layout()
plt.show()

# RMSE
train_rmse_history = [np.sqrt(m) for m in train_mse_history]
val_rmse_history   = [np.sqrt(m) for m in val_mse_history]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(train_rmse_history, 'r', label='train RMSE')
ax.plot(val_rmse_history, 'b', label='val RMSE')
ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
ax.legend()
ax.set_title('Evolución RMSE por epoch (NN - MLPRegressor)', fontsize=16)
ax.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Dispersión real vs estimado (train y test), como en los otros modelos
# -------------------------------------------------------------------------
# Train
xmin, xmax = np.min(ytrain_nn), np.max(ytrain_nn)
xline = np.linspace(xmin, xmax)
plt.figure(figsize=(10, 6))
plt.scatter(ytrain_nn, yhat_tr, label='Estimación')
plt.plot(xline, xline, 'k--', label='Estimación perfecta')
plt.xlabel('Valor real (train)', fontsize=16)
plt.ylabel('Valor estimado', fontsize=16)
plt.title('NN (MLP) - R² train = %.4f' % R2_nn_train, fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Test
xmin, xmax = np.min(ytest_nn), np.max(ytest_nn)
xline = np.linspace(xmin, xmax)
plt.figure(figsize=(10, 6))
plt.scatter(ytest_nn, yhat_ts, label='Estimación')
plt.plot(xline, xline, 'k--', label='Estimación perfecta')
plt.xlabel('Valor real (test)', fontsize=16)
plt.ylabel('Valor estimado', fontsize=16)
plt.title('NN (MLP) - R² test = %.4f' % R2_nn_test, fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# MODELOS 5 y 6: Ridge y Lasso
# =============================================================================

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold

# Usamos ytrain en 1D para los modelos lineales
ytrain_lin = ytrain.ravel()

# Rejilla de hiperparámetros
alphas = [1000, 500, 200, 100, 50, 20, 10, 1, 0.1, 0.01]

folds = KFold(n_splits=10, shuffle=True, random_state=42)

# ---------------- Ridge ----------------
ridge = Ridge()
ridge_param_grid = {'alpha': alphas}

ridge_grid = GridSearchCV(
    estimator=ridge,
    param_grid=ridge_param_grid,
    scoring='r2',
    cv=folds,
    n_jobs=-1
)

ridge_grid.fit(xtrain, ytrain_lin)
ridge_best = ridge_grid.best_estimator_

print("\n=== Ridge Regression ===")
print("Mejores hiperparámetros:", ridge_grid.best_params_)
print("R² train (cv):", ridge_grid.best_score_)

# =============================================================================
# INTERPRETACIÓN DE RIDGE: COEFICIENTES POR VARIABLE
# =============================================================================

# Coeficientes en el espacio ESCALADO
coef_scaled = ridge_best.coef_.ravel()  # shape: (n_features,)

# std_x y std_y usados por los scalers
std_X = scaler_X.scale_                   # array de std por feature
std_Y = scaler_Y.scale_[0]               # std de la variable objetivo

# Transformación aproximada a unidades originales:
# Si entrenamos sobre:
#   X_std = (X - mean_X) / std_X
#   y_std = (y - mean_Y) / std_Y
# entonces, en unidades originales:
#   coef_original ≈ coef_scaled * (std_Y / std_X)
coef_original = coef_scaled * (std_Y / std_X)

ridge_coefs = (
    pd.DataFrame({
        'feature': feature_names,
        'coef_scaled': coef_scaled,
        'coef_original_y_units': coef_original,
        'abs_coef_scaled': np.abs(coef_scaled)
    })
    .sort_values('abs_coef_scaled', ascending=False)
    .reset_index(drop=True)
)

print("\nTop 15 variables según Ridge (por |coeficiente escalado|):")
print(ridge_coefs.head(15))

# Gráfico para la tesis: importancia (absoluta) de los coeficientes de Ridge
top_k = 15
plt.figure(figsize=(10, 6))
plt.barh(
    ridge_coefs['feature'][:top_k][::-1],
    ridge_coefs['abs_coef_scaled'][:top_k][::-1]
)
plt.xlabel('|coeficiente (escala estandarizada)|', fontsize=12)
plt.title('Importancia de variables según Ridge', fontsize=14)
plt.tight_layout()
plt.show()


# ---------------- Lasso ----------------
lasso = Lasso(max_iter=10000)
lasso_param_grid = {'alpha': alphas}

lasso_grid = GridSearchCV(
    estimator=lasso,
    param_grid=lasso_param_grid,
    scoring='r2',
    cv=folds,
    n_jobs=-1
)

lasso_grid.fit(xtrain, ytrain_lin)
lasso_best = lasso_grid.best_estimator_

print("\n=== Lasso Regression ===")
print("Mejores hiperparámetros:", lasso_grid.best_params_)
print("R² train (cv):", lasso_grid.best_score_)


#%%
# =============================================================================
# COMPARATIVO FINAL DE MODELOS (GENÉRICO, CON MÉTRICAS EN PESOS REALES)
# =============================================================================

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

def resumen_modelo(nombre, modelo, xtr, ytr, xte, yte, scaler_Y):
    est = modelo.best_estimator_ if hasattr(modelo, "best_estimator_") else modelo

    ytr_true = ytr.ravel()
    yte_true = yte.ravel()

    ytr_pred = est.predict(xtr)
    yte_pred = est.predict(xte)

    # Métricas en escala estandarizada
    r2_tr = r2_score(ytr_true, ytr_pred)
    r2_te = r2_score(yte_true, yte_pred)

    rmse_tr = root_mean_squared_error(ytr_true, ytr_pred)
    rmse_te = root_mean_squared_error(yte_true, yte_pred)

    mae_tr = mean_absolute_error(ytr_true, ytr_pred)
    mae_te = mean_absolute_error(yte_true, yte_pred)

    # Conversión a pesos reales
    ytr_true_real = scaler_Y.inverse_transform(ytr.reshape(-1, 1)).ravel()
    yte_true_real = scaler_Y.inverse_transform(yte.reshape(-1, 1)).ravel()

    ytr_pred_real = scaler_Y.inverse_transform(ytr_pred.reshape(-1, 1)).ravel()
    yte_pred_real = scaler_Y.inverse_transform(yte_pred.reshape(-1, 1)).ravel()

    rmse_tr_real = root_mean_squared_error(ytr_true_real, ytr_pred_real)
    rmse_te_real = root_mean_squared_error(yte_true_real, yte_pred_real)

    mae_tr_real = mean_absolute_error(ytr_true_real, ytr_pred_real)
    mae_te_real = mean_absolute_error(yte_true_real, yte_pred_real)

    return {
        "Modelo": nombre,
        "R2_train": r2_tr,
        "R2_test": r2_te,
        "RMSE_train_scaled": rmse_tr,
        "RMSE_test_scaled": rmse_te,
        "MAE_train_scaled": mae_tr,
        "MAE_test_scaled": mae_te,
        "RMSE_train_$": rmse_tr_real,
        "RMSE_test_$": rmse_te_real,
        "MAE_train_$": mae_tr_real,
        "MAE_test_$": mae_te_real,
        "Gap_R2_train_test": r2_tr - r2_te
    }

def construir_resumen_modelos(modelos, xtrain, ytrain, xtest, ytest, scaler_Y):
    filas = []
    for nombre, modelo in modelos:
        filas.append(resumen_modelo(nombre, modelo, xtrain, ytrain, xtest, ytest, scaler_Y))
    df = pd.DataFrame(filas).set_index("Modelo")
    return df

# Lista de modelos a evaluar
modelos_a_evaluar = [
    ("Linear Regression",       lr_model),
    ("Decision Tree",           model2),
    ("Bagging Trees (tuned)",   bt),
    ("Neural Network (MLP)",    mlp),
    ("Ridge Regression",        ridge_best),
    ("Lasso Regression",        lasso_best),
]

resultados_modelos = construir_resumen_modelos(
    modelos_a_evaluar, xtrain, ytrain, xtest, ytest, scaler_Y
)

print("\n================= RESUMEN COMPARATIVO DE MODELOS =================")
print(resultados_modelos.round(4))

# Ranking por R² en test, mostrando error real en pesos
ranking_r2 = resultados_modelos.sort_values("R2_test", ascending=False)[
    ["R2_test", "RMSE_test_$", "MAE_test_$", "Gap_R2_train_test"]
]
print("\nRanking por R² en test (mejor arriba):")
print(ranking_r2.round(2))

# -----------------------------------------------------------------------------
# Gráfica: R² en test por modelo (lista para incluir en tesis)
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
ranking_r2["R2_test"].plot(kind="bar")
plt.ylabel("R² en conjunto de prueba", fontsize=14)
plt.title("Comparación de R² en conjunto de prueba", fontsize=16)
plt.xticks(rotation=25, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Gráfica: Gap de R²
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
resultados_modelos["Gap_R2_train_test"].plot(kind="bar", color="orange")
plt.axhline(0, color='k', linewidth=1)
plt.ylabel("R²_train - R²_test", fontsize=14)
plt.title("Indicador de sobreajuste por modelo", fontsize=16)
plt.xticks(rotation=25, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =============================================================================
# GUARDADO DE ARTEFACTOS DEL MODELO FINAL
# =============================================================================

import joblib
import json

# CAMBIO:
# Ajusta aquí el modelo final que decidas congelar para despliegue.
# Recomendación actual: ridge_best o lr_model
modelo_final = ridge_best

joblib.dump(modelo_final, "modelo_final.pkl")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_Y, "scaler_Y.pkl")

with open("feature_names.json", "w", encoding="utf-8") as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=4)

print("Artefactos guardados correctamente.")