# Estadistica descriptiva variables continuas

#Librerias
#Graficos 
import matplotlib.pyplot as plt
import seaborn as sns

#Procesado
import pandas as pd
#------------------------------------------
def convert_to_seconds(delta):
    total_seconds = delta.total_seconds()
    seconds = int(total_seconds)
    return seconds
#------------------------------------------

# MAIN
prueba = pd.read_excel('prueba.xlsx')
prueba['tiempo'] = pd.to_timedelta(prueba['tiempo'])
prueba = prueba[prueba['dias']<10]
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
prueba.rename(columns={'gruas':'Gruas','velmedia':'Viento'}, inplace=True)

# Graficando
variables_continuas = ['GT', 'Eslora', 'Manga','Calado_Popa_Entrada','Calado_Popa_Salida', 'Viento']
# Gráfico de distribución para cada variable numérica
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 5))
axes = axes.flat


for i, colum in enumerate(prueba[variables_continuas]):
    sns.histplot(
        data     = prueba,
        x        = colum,
        stat     = "count",
        alpha    = 0.3,
        ax       = axes[i]
    )
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].set_ylabel("Nº de observaciones", fontsize = 5)
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
  
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Histograma variables continuas', fontsize = 10, fontweight = "bold")
plt.show()

# BOXPLOTS
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 5))
axes = axes.flat


for i, colum in enumerate(prueba[variables_continuas]):
    sns.boxplot(
        data     = prueba,
        y        = colum,
        ax       = axes[i]
    )
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].set_ylabel("")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
  
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Box-plots variables continuas', fontsize = 10, fontweight = "bold")
plt.show()

# Medidas descriptivas
print(prueba[variables_continuas].describe())