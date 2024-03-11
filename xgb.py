# Aplicacion y optimizacion hiperparametros XGBOOST

#Librerias
# ==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedKFold, RandomizedSearchCV
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
discretas = ['Gruas', 'Semana','Mes', 'DiaSemana','DiaAnyo']
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

from xgboost import XGBRegressor
# ==============================================================================
# param_grid = {'max_depth'        : [None, 2, 3, 4, 5, 6],
#               'min_child_weight' : [1,3,5,7,9],
#               'learning_rate'    : [0.01, 0.05, 0.1],
#               'n_estimators'     : [1000,2000,3000,4000,5000]
#              }
# # XGBoost necesita pasar los paramétros específicos del entrenamiento al llamar
# # al método .fit()
# fit_params = {
#               "eval_set": [(X_test, y_test)],
#               "verbose": False
#              }
# grid = GridSearchCV(
#         estimator  = XGBRegressor(
#                         early_stopping_rounds = 5,
#                         eval_metric           = "mae",
#                         random_state          = 123
#                     ),
#         param_grid = param_grid,
#         scoring    = 'neg_root_mean_squared_error',
#         n_jobs     = -1,
#         cv         = RepeatedKFold(n_splits=3, n_repeats=1, random_state=123), 
#         refit      = True,
#         verbose    = 0,
#         return_train_score = True
#        )

# grid.fit(X = X_train, y = y_train, **fit_params)

# # Resultados
# # ==============================================================================
# resultados = pd.DataFrame(grid.cv_results_)
# print(resultados.filter(regex = '(param.*|mean_t|std_t)') \
#     .drop(columns = 'params') \
#     .sort_values('mean_test_score', ascending = False) \
#     .head(4)
# )
xgb = XGBRegressor(
                        n_estimators          = 5000,
                        early_stopping_rounds = 5,
                        eval_metric           = "mae",
                        random_state          = 123,
                        learning_rate = 0.1,
                        max_depth = 6, 
                        subsample = 1.0,
                        min_child_weight = 9
)
xgb.fit(X_train,
        y_train, 
        eval_set = [(X_test,y_test)],
        verbose = False
        )
predicciones = xgb.predict(X = X_test)
rmse_xgb = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
          )
mae_xgb = mean_absolute_error(y_test, predicciones)
print(f"El error (rmse) de test es: {rmse_xgb}, {rmse_xgb/3600} y {rmse_xgb/(3600*24)}")
print(f"El error (mae) de test es: {mae_xgb}, {mae_xgb/3600} y {mae_xgb/(3600*24)}")