# Estadistica descriptiva, normalidad

#Librerias
# ==============================================================================
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
#Graficos 
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D

#Procesado
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from gower import gower_matrix 

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')
#------------------------------------------
def convert_to_seconds(delta):
    total_seconds = delta.total_seconds()
    seconds = int(total_seconds)
    return seconds

def escalar_fechas(date_str):
    # date = datetime.strptime(date_str,"%Y-%m-%d")
    escala = int(date_str.strftime("%Y%m%d"))
    return escala

def rango_fechas(desde, hasta):
    return [desde + relativedelta(days=days) for days in range((hasta - desde).days + 1)]

def gower_distance(point1, point2, categorical_weight=1.0):
    """
    Calcula la distancia de Gower entre dos puntos.

    Args:
    - point1: numpy array, primer punto
    - point2: numpy array, segundo punto
    - categorical_weight: float, peso para las variables categóricas (0 <= categorical_weight <= 1)

    Returns:
    - distance: float, distancia de Gower entre los dos puntos
    """
    n = len(point1)
    sum_s = 0.0
    sum_w = 0.0
    
    for i in range(n):
        if isinstance(point1[i], int) and isinstance(point2[i], int):
            # Si ambas variables son categóricas
            if point1[i] == point2[i]:
                sum_s += 0.0
            else:
                sum_s += 1.0
            sum_w += categorical_weight
        else:
            # Si al menos una de las variables es numérica
            if point1[i] != point2[i]:
                sum_s += np.abs(point1[i] - point2[i])
            sum_w += 1.0
    
    distance = sum_s / sum_w
    
    return distance
#------------------------------------------
# MAIN
prueba = pd.read_excel('prueba.xlsx')
prueba.dropna(inplace=True)
prueba['tiempo'] = pd.to_timedelta(prueba['tiempo'])
prueba['seconds'] = prueba['tiempo'].apply(convert_to_seconds)
prueba['DiaSemana'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.day_of_week
prueba['Mes'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.month
prueba['DiaAnyo'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.day_of_year
prueba['Semana'] = (prueba['Hora_Inicio_Servicio_Entrada'].dt.isocalendar().week).astype('int64')
prueba['Hora'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.hour
# Incluyendo vientos medios
viento = pd.read_excel('Viento_Valencia_Viveros.xlsx')

prueba["fecha"] = prueba["Hora_Inicio_Servicio_Entrada"].dt.date
prueba["fin"] = prueba["Hora_Fin_Servicio_Salida"].dt.date
prueba['fecha'] = pd.to_datetime(prueba['fecha'], format='%d/%m/%Y')
prueba = prueba.merge(viento, on='fecha')
prueba.rename(columns={'velmedia':'Viento'}, inplace=True)

# # Nueva GRUAS
prueba['gruas'] = prueba['gruas'].astype('category')
# Define una función para reagrupar las categorías
def reagrupar_categoria(categoria):
    if categoria <= 2:
        return categoria
    else:
        return "3"

prueba['Gruas'] = (prueba['gruas'].map(reagrupar_categoria)).astype('int32')


variables_X = ['GT', 'Eslora', 'Manga','Calado_Popa_Entrada','Calado_Popa_Salida','Gruas','DockCode','Semana','Mes', 'DiaSemana','Viento']
X = prueba[variables_X]
y = prueba['seconds']

# Construyendo el preprocesador 
continuas_prep = Pipeline(
                        steps=[
                            ('scaler', StandardScaler())
                        ]
                      )
categoricas_prep = Pipeline(
                            steps=[
                                ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False))
                            ]
                          )
discretas_prep = Pipeline(
    steps=[('ordinal', OrdinalEncoder())]
)


# Preprocesador

continuas = ['GT', 'Eslora', 'Manga','Calado_Popa_Entrada','Calado_Popa_Salida', 'Viento']
discretas = ['Gruas', 'Semana','Mes', 'DiaSemana']
categorica = ['DockCode']
preprocessor = ColumnTransformer(
                    transformers=[
                        ('continuas', continuas_prep, continuas),
                        ('categorica', categoricas_prep, categorica),
                        ('discretas', discretas_prep, discretas)
                    ],
                    remainder='passthrough',
                    verbose_feature_names_out = False
               ).set_output(transform="pandas")

# #CLUSTER 
from gower import gower_matrix

X_final = preprocessor.fit_transform(X)
X_matrix = gower_matrix(X_final)

# LOF
from sklearn.neighbors import LocalOutlierFactor

labels_y = y[y>400000]
print(X.index)
lof = LocalOutlierFactor(n_neighbors=27, metric='precomputed')
y_pred = lof.fit_predict(X_matrix)
X_scores = -1*lof.negative_outlier_factor_
lof_X = pd.DataFrame({'LOF':X_scores},index = X.index)
print(lof_X[lof_X['LOF']>1.4])
#Histograma
# handle = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)]
# label_lof = ['Y > 400000']
# fig, ax = plt.subplots()
# sns.histplot(
#         data     = X_final,
#         x        = X_scores,
#         stat     = "count",
#         alpha    = 0.3,
#         ax       = ax
#     )
# for i in labels_y.index:
#     print(f'{X_scores[i]} y {y[i]}')
#     sns.scatterplot(x=X_scores[i], y=np.ones_like(labels_y), color='red', s=100)
# ax.set_xlabel('LOF')
# ax.set_ylabel('Nº de observaciones')
# ax.tick_params(labelsize = 6)
# fig.suptitle('Histograma LOF', fontsize = 10, fontweight = "bold")
# plt.legend(handle,label_lof, loc='center right')
# plt.show()


# # MDS + LOF

# # plt.scatter(embedded_data[:, 0], embedded_data[:, 1], color="k", s=3.0, label="Data points")
# # # plot circles with radius proportional to the outlier scores
# # radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
# # scatter = plt.scatter(
# #     embedded_data[:, 0],
# #     embedded_data[:, 1],
# #     s=1000 * radius,
# #     edgecolors="r",
# #     facecolors="none",
# #     label="Outlier scores",
# # )
# # plt.axis("tight")
# # plt.title("Local Outlier Factor (LOF)")
# # plt.show()

# # # MDS
# from sklearn.manifold import MDS
# mds = MDS(n_components=2,metric=True,dissimilarity='precomputed')
# embedded_data = mds.fit_transform(X_matrix)
# colors = {32: 'red', 36: 'blue', 75: 'green', 76: 'orange'}
# colors_dark = {32: '#940102', 36: '#040166', 75: '#174604', 76: '#875202'}
# # Crear una leyenda personalizada
# handles = []
# labels_1 = []
# for value, color in colors.items():
#     handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
#     labels_1.append(str(value))

# #Visualizar los datos en 2D
# # 1 - SIN ETIQUETAS  
# plt.scatter(embedded_data[:, 0], embedded_data[:, 1], color = '#409cd5')
# plt.title('Visualización MDS en 2D')
# plt.xlabel('Dimensión 1')
# plt.ylabel('Dimensión 2')
# plt.show()

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (10,5))
# axes = axes.flat
# # 2 - ETIQUETAS A OBSERVACIONES > 400000 SEGUNDOS
# # Añadir etiquetas a cada punto
# labels_y = y[y>400000]
# axes[0].scatter(embedded_data[:, 0], embedded_data[:, 1], color = '#409cd5')
# for i in labels_y.index:
#     axes[0].scatter(embedded_data[i, 0], embedded_data[i, 1],  color = '#060270')
# axes[0].set_xlabel('Dimensión 1')
# axes[0].set_ylabel('Dimensión 2')
# # 3 - ETIQUETAS POR DOCKCODE
# labels_dockcode = X['DockCode']
# for i, label in enumerate(labels_dockcode):
#     axes[1].scatter(embedded_data[i,0], embedded_data[i,1],c=colors[label])
# for i in labels_y.index:
#     axes[1].scatter(embedded_data[i, 0], embedded_data[i, 1],  c = colors_dark[X.at[i,'DockCode']])
# axes[1].legend(handles, labels_1, title='DockCode')

# axes[1].set_xlabel('Dimensión 1')
# axes[1].set_ylabel('Dimensión 2')

# fig.tight_layout()
# plt.subplots_adjust(top = 0.9)
# fig.suptitle('MDS - 2D', fontsize = 10, fontweight = "bold")
# plt.show()

# # 4 - Pasos 2 y 3 con LOF > 1.4

# fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize = (10,5))
# axes1 = axes1.flat
# # 2 - ETIQUETAS A OBSERVACIONES LOF > 1.4
# # Añadir etiquetas a cada punto
labels_X = lof_X[lof_X['LOF']>1.4]
# axes1[0].scatter(embedded_data[:, 0], embedded_data[:, 1], color = '#409cd5')
seconds_lof =[]
for i in labels_X.index:
    seconds_lof.append(y.iloc[i])
labels_X['seconds'] = seconds_lof
print(labels_X)
#     axes1[0].scatter(embedded_data[i, 0], embedded_data[i, 1],  color = '#060270')
# axes1[0].set_xlabel('Dimensión 1')
# axes1[0].set_ylabel('Dimensión 2')
# # 3 - ETIQUETAS POR DOCKCODE y LOF > 1.4
# labels_dockcode = X['DockCode']
# for i, label in enumerate(labels_dockcode):
#     axes1[1].scatter(embedded_data[i,0], embedded_data[i,1],c=colors[label])
# for i in labels_X.index:
#     axes1[1].scatter(embedded_data[i, 0], embedded_data[i, 1],  c = colors_dark[X.at[i,'DockCode']])
# axes1[1].legend(handles, labels_1, title='DockCode')

# axes1[1].set_xlabel('Dimensión 1')
# axes1[1].set_ylabel('Dimensión 2')

# fig1.tight_layout()
# plt.subplots_adjust(top = 0.9)
# fig1.suptitle('MDS - 2D', fontsize = 10, fontweight = "bold")
# plt.show()

# # ISOLATION FOREST
# from sklearn.ensemble import IsolationForest
# variables_X = ['GT', 'Eslora', 'Manga','Calado_Popa_Entrada','Calado_Popa_Salida','Gruas', 'DockCode','Semana','Mes', 'DiaSemana','Viento']
# X = prueba[variables_X]
# y = prueba['seconds']
# labels_y = y[y>400000]

# X_muelle = X[X['DockCode']==36]
# X_filtrado_muelle = preprocessor.fit_transform(X_muelle)
# iso = IsolationForest(n_estimators=5000,max_samples=25, random_state=0)
# iso.fit_predict(X_final)
# predicciones = -iso.score_samples(X_final)
# print(predicciones.min())
# print(predicciones.max())

# #Histograma
# handle = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)]
# label_lof = ['Y > 400000']
# fig, ax = plt.subplots()
# sns.histplot(
#         data     = X,
#         x        = predicciones,
#         stat     = "count",
#         alpha    = 0.3,
#         ax       = ax
#     )
# for i in labels_y.index:
#     # print(f'{predicciones[i]} y {y[i]}')
#     sns.scatterplot(x=predicciones[i], y=np.ones_like(labels_y), color='red', s=100)
# ax.set_xlabel('Anomaly Score')
# ax.set_ylabel('Nº de observaciones')
# ax.tick_params(labelsize = 6)
# fig.suptitle('Histograma Isolation Forest', fontsize = 10, fontweight = "bold")
# plt.legend(handle,label_lof, loc='center right')
# plt.show()

