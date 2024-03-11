# Estadistica descriptiva, normalidad

#Librerias
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
#------------------------------------------
def convert_to_seconds(delta):
    total_seconds = delta.total_seconds()
    seconds = int(total_seconds)
    return seconds
#------------------------------------------
# MAIN
prueba = pd.read_excel('prueba.xlsx')
prueba = prueba[prueba['dias']<10]
prueba['tiempo'] = pd.to_timedelta(prueba['tiempo'])
prueba['seconds'] = prueba['tiempo'].apply(convert_to_seconds)
prueba['dia_semana'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.day_of_week
prueba['mes'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.month
prueba['dia_ano'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.day_of_year

# Incluyendo vientos medios
viento = pd.read_excel('Viento_Valencia_Viveros.xlsx')

prueba["fecha"] = prueba["Hora_Inicio_Servicio_Entrada"].dt.date
prueba["fin"] = prueba["Hora_Fin_Servicio_Salida"].dt.date
prueba['fecha'] = pd.to_datetime(prueba['fecha'], format='%d/%m/%Y')
prueba = prueba.merge(viento, on='fecha')
prueba.dropna(inplace=True)


# Graficando
#Histograma
fig, ax = plt.subplots()
sns.histplot(
        data     = prueba,
        x        = prueba['seconds'],
        stat     = "count",
        alpha    = 0.3,
        ax       = ax
    )
ax.set_xlabel('Seconds')
ax.set_ylabel('Nº de observaciones')
ax.tick_params(labelsize = 6)
fig.suptitle('Histograma Y', fontsize = 10, fontweight = "bold")
plt.show()

#Medidas descriptivas
print((prueba['seconds'].describe()))