# Estadistica descriptiva, normalidad

#Librerias
# ==============================================================================
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgb import XGBRegressor
#Graficos 
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm

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
#------------------------------------------
# MAIN
prueba = pd.read_excel('prueba.xlsx')
prueba['tiempo'] = pd.to_timedelta(prueba['tiempo'])
print(prueba.shape)
prueba = prueba[prueba['dias']<10]
print(prueba['DockCode'].describe())
prueba['seconds'] = prueba['tiempo'].apply(convert_to_seconds)
prueba['DiaSemana'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.day_of_week
prueba['Mes'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.month
prueba['DiaAnyo'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.day_of_year
prueba['Semana'] = (prueba['Hora_Inicio_Servicio_Entrada'].dt.isocalendar().week).astype('int64')

# Incluyendo vientos medios
viento = pd.read_excel('Viento_Valencia_Viveros.xlsx')

prueba["fecha"] = prueba["Hora_Inicio_Servicio_Entrada"].dt.date
prueba["fin"] = prueba["Hora_Fin_Servicio_Salida"].dt.date
prueba['fecha'] = pd.to_datetime(prueba['fecha'], format='%d/%m/%Y')
prueba = prueba.merge(viento, on='fecha')
prueba.dropna(inplace=True)
print(prueba.shape)
prueba.rename(columns={'velmedia':'Viento'}, inplace=True)

# Nueva GRUAS
prueba['gruas'] = prueba['gruas'].astype('category')
# Define una función para reagrupar las categorías
def reagrupar_categoria(categoria):
    if categoria <= 2:
        return categoria
    else:
        return "3"

prueba['Gruas'] = (prueba['gruas'].map(reagrupar_categoria)).astype('int32')


variables_X = ['GT', 'Eslora', 'Manga','Calado_Popa_Entrada','Calado_Popa_Salida','Gruas', 'DockCode', 'Semana','Mes', 'DiaSemana','Viento']
X = prueba[variables_X]
y = prueba['seconds']
X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
print(X.dtypes)
# Construyendo el preprocesador 
continuas_prep = Pipeline(
                        steps=[
                            ('scaler', StandardScaler())
                        ]
                      )
categoricas_prep = Pipeline(
                            steps=[
                                ('onehot', OneHotEncoder(sparse_output=False))
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
                        ('categoricas', categoricas_prep, categorica),
                        ('discretas', discretas_prep, discretas)
                    ],
                    remainder='passthrough',
                    verbose_feature_names_out = False
               ).set_output(transform="pandas")

#===============================
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances

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


#===============================
# Calcular la matriz de distancias
distance_matrix_train = gower_matrix(X_train)
distance_matrix_test = gower_matrix(X_test).T

from sklearn.neighbors import KNeighborsRegressor 
knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsRegressor(metric='braycurtis'))
    ])
# knn_pipeline.fit(X_train,y_train)
# preds = knn_pipeline.predict(X_test)
# print( mean_absolute_error(y_test, preds))
# print( mean_squared_error(y_test, preds, squared=False))

# Espacio de búsqueda de cada hiperparámetro
param_distributions = {'model__n_neighbors': np.linspace(1, 100, 500, dtype=int)}

# Búsqueda random grid
grid = RandomizedSearchCV(
        estimator  = knn_pipeline,
        param_distributions = param_distributions,
        n_iter     = 20,
        scoring    = 'neg_root_mean_squared_error',
        n_jobs     = -1,
        cv         = RepeatedKFold(n_splits = 5, n_repeats = 3), 
        refit      = True, 
        verbose    = 0,
        random_state = 123,
        return_train_score = True
       )

grid.fit(X = X_train, y = y_train)

# Resultados del grid
# ==============================================================================
resultados = pd.DataFrame(grid.cv_results_)
print(resultados.filter(regex = '(param.*|mean_t|std_t)')\
    .drop(columns = 'params')\
    .sort_values('mean_test_score', ascending = False)\
    .head(1)
)
# Error de test del modelo final
# ==============================================================================
modelo_final = grid.best_estimator_
predicciones = modelo_final.predict(X = X_test)
rmse_knn = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
mae_knn = mean_absolute_error(y_test, predicciones)
print(f"El error (rmse) de test es: {rmse_knn}")
print(f"El error (mae) de test es: {mae_knn}")
