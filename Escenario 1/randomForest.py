# Aplicacion y optimizacion hiperparametros RANDOM FOREST

#Librerias
# ==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Procesado
import pandas as pd
from dateutil.relativedelta import relativedelta

#------------------------------------------
def convert_to_seconds(delta):
    total_seconds = delta.total_seconds()
    seconds = int(total_seconds)
    return seconds

def rango_fechas(desde, hasta):
    return [desde + relativedelta(days=days) for days in range((hasta - desde).days + 1)]
#----------------------------------------
# MAIN
prueba = pd.read_excel('prueba.xlsx')
prueba['tiempo'] = pd.to_timedelta(prueba['tiempo'])
prueba = prueba[prueba['dias']<10]
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


variables_X = ['GT', 'Eslora', 'Manga','Calado_Popa_Entrada','Calado_Popa_Salida','Gruas', 'DockCode','Semana','Mes', 'DiaSemana','Viento']
X = prueba[variables_X]
y = prueba['seconds']
X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
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

# Se combinan los pasos de preprocesado y el modelo en un mismo pipeline.
from sklearn.ensemble import RandomForestRegressor
pipe = Pipeline([('preprocessing', preprocessor),
                 ('modelo', RandomForestRegressor(n_estimators=2000,max_features=1.0,max_depth=10, criterion='absolute_error'))])

# Optimización de hiperparámetros
# ==============================================================================
# Espacio de búsqueda de cada hiperparámetro

# param_distributions = {
#     'modelo__n_estimators': [50, 100, 1000, 2000],
#     'modelo__max_features': [3, 5, 7, 1.0],
#     'modelo__max_depth'   : [None, 3, 5, 10, 20]
# }

# # Búsqueda random grid
# grid = RandomizedSearchCV(
#         estimator  = pipe,
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
#     .head(1))
# Error de test del modelo final
# ==============================================================================
pipe.fit(X_train,y_train)
predicciones = pipe.predict(X = X_test)
cv_scores = cross_val_score(
                estimator = pipe,
                X         = X_train,
                y         = y_train,
                scoring   = 'neg_mean_absolute_error',
                cv        = 5
            )
cv_rf = -1*cv_scores.mean()
rmse_rf = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
          )
mae_rf = mean_absolute_error(y_test, predicciones)
print(f"El error (cv) de test es: {cv_rf}, {cv_rf/3600} y {cv_rf/(3600*24)}")
print(f"El error (rmse) de test es: {rmse_rf}, {rmse_rf/3600} y {rmse_rf/(3600*24)}")
print(f"El error (mae) de test es: {mae_rf}, {mae_rf/3600} y {mae_rf/(3600*24)}")
