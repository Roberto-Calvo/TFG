# Aplicacion y optimizacion hiperparametros KNN

#Librerias
# =============================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Graficos 
import matplotlib.pyplot as plt
from matplotlib import style

#Procesado
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

#-----------------------------------------
def convert_to_seconds(delta):
    total_seconds = delta.total_seconds()
    seconds = int(total_seconds)
    return seconds

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

def gower_distance1(point1, point2, categorical_weights=None):
    """
    Calcula la distancia de Gower entre dos puntos.

    Args:
    - point1 (list, tuple, numpy array): Coordenadas del primer punto.
    - point2 (list, tuple, numpy array): Coordenadas del segundo punto.
    - categorical_weights (list, numpy array, optional): Pesos para atributos categóricos.

    Returns:
    - distance (float): Distancia de Gower entre los dos puntos.
    """

    point1 = np.array(point1)
    point2 = np.array(point2)

    if categorical_weights is None:
        categorical_weights = np.ones(len(point1))

    # Identificar índices de atributos categóricos
    categorical_indices = np.where(categorical_weights != 0)[0]

    # Calcular distancia para cada atributo
    distances = []
    for i, (attr1, attr2, weight) in enumerate(zip(point1, point2, categorical_weights)):
        if i in categorical_indices:
            # Si es un atributo categórico, la distancia es 0 si son iguales, 1 si son diferentes
            distance = 0 if attr1 == attr2 else 1
        else:
            # Si es un atributo numérico, calcular la distancia normalizada
            range_attr = np.max(attr1) - np.min(attr1)
            if range_attr == 0:
                distance = 0 if attr1 == attr2 else 1
            else:
                distance = np.abs(attr1 - attr2) / range_attr
        distances.append(distance * weight)

    # Calcular la distancia promedio
    distance = np.sum(distances) / np.sum(categorical_weights)

    return distance

#===============================

from sklearn.neighbors import KNeighborsRegressor 
knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsRegressor(n_neighbors=27,metric=gower_distance1))
    ])
knn_pipeline.fit(X_train,y_train)
predicciones = knn_pipeline.predict(X_test)
rmse_knn = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
          )
mae_knn = mean_absolute_error(y_test, predicciones)
print(f"El error (rmse) de test es: {rmse_knn}, {rmse_knn/3600} y {rmse_knn/(3600*24)}")
print(f"El error (mae) de test es: {mae_knn}, {mae_knn/3600} y {mae_knn/(3600*24)}")

# # Espacio de búsqueda de cada hiperparámetro
# param_distributions = {'model__n_neighbors': np.linspace(1, 100, 500, dtype=int)}

# # Búsqueda random grid
# grid = RandomizedSearchCV(
#         estimator  = knn_pipeline,
#         param_distributions = param_distributions,
#         n_iter     = 20,
#         scoring    = 'neg_root_mean_squared_error',
#         n_jobs     = -1,
#         cv         = RepeatedKFold(n_splits = 5, n_repeats = 3), 
#         refit      = True, 
#         verbose    = 0,
#         random_state = 123,
#         return_train_score = True
#        )

# grid.fit(X = X_train, y = y_train)

# # Resultados del grid
# # ==============================================================================
# resultados = pd.DataFrame(grid.cv_results_)
# print(resultados.filter(regex = '(param.*|mean_t|std_t)')\
#     .drop(columns = 'params')\
#     .sort_values('mean_test_score', ascending = False)\
#     .head(1)
# )
# # Error de test del modelo final
# # ==============================================================================
# modelo_final = grid.best_estimator_
# predicciones = modelo_final.predict(X = X_test)
# rmse_knn = mean_squared_error(
#             y_true  = y_test,
#             y_pred  = predicciones,
#             squared = False
#            )
# mae_knn = mean_absolute_error(y_test, predicciones)
# print(f"El error (rmse) de test es: {rmse_knn}")
# print(f"El error (mae) de test es: {mae_knn}")
