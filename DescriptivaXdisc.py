# Estadistica descriptiva de las variables discretas

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
print(prueba.shape)
prueba = prueba[prueba['dias']<10]
print(prueba['DockCode'].describe())
prueba['seconds'] = prueba['tiempo'].apply(convert_to_seconds)
prueba['DiaSemana'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.day_of_week
prueba['Mes'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.month
prueba['DiaAnyo'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.day_of_year
prueba['Semana'] = prueba['Hora_Inicio_Servicio_Entrada'].dt.isocalendar().week

# Incluyendo vientos medios
viento = pd.read_excel('Viento_Valencia_Viveros.xlsx')

prueba["fecha"] = prueba["Hora_Inicio_Servicio_Entrada"].dt.date
prueba["fin"] = prueba["Hora_Fin_Servicio_Salida"].dt.date
prueba['fecha'] = pd.to_datetime(prueba['fecha'], format='%d/%m/%Y')
prueba = prueba.merge(viento, on='fecha')
prueba.dropna(inplace=True)
print(prueba.shape)
prueba.rename(columns={'gruas':'Gruas','velmedia':'Viento'}, inplace=True)
prueba['DockCode'] = prueba['DockCode'].astype('category')

variables_discretas = ['DockCode',  'Mes', 'DiaSemana', 'Gruas']

# Gráfico para cada variable cualitativa
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))
axes = axes.flat

for i, colum in enumerate(prueba[variables_discretas]):
    sns.countplot(
        data=prueba,
        x=colum,
        alpha=0.5,
        edgecolor = 'black',
        ax=axes[i]
    )
    axes[i].set_title(colum, fontsize=7, fontweight="bold")
    axes[i].set_ylabel("Nº de observaciones", fontsize=5)
    axes[i].tick_params(labelsize=6)
    axes[i].set_xlabel("")
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Gráficos de Barras para Variables Discretas', fontsize=10, fontweight="bold")
plt.show()
print(prueba.shape)

# SEMANA_ANYO
fig, ax = plt.subplots(figsize = (18,5))
sns.countplot(
        data     = prueba,
        x        = prueba['Semana'],
        alpha    = 0.5,
        edgecolor = 'black',
        ax       = ax
    )
ax.set_xlabel('Semana')
ax.set_ylabel('Nº de observaciones')
ax.tick_params(labelsize = 6)
fig.suptitle('Gráfico de Barras Semana', fontsize = 10, fontweight = "bold")
plt.show()

# Nueva GRUAS
prueba['Gruas'] = prueba['Gruas'].astype('category')
# Define una función para reagrupar las categorías
def reagrupar_categoria(categoria):
    if categoria <= 2:
        return categoria
    else:
        return ">2"


prueba['Gruas_regrup'] = prueba['Gruas'].map(reagrupar_categoria)
print(prueba['Gruas_regrup'].describe())
fig, ax = plt.subplots(figsize = (4,4))
sns.countplot(
        data     = prueba,
        x        = prueba['Gruas_regrup'],
        alpha    = 0.5,
        edgecolor = 'black',
        ax       = ax
    )
ax.set_xlabel('Gruas')
ax.set_ylabel('Nº de observaciones')
ax.tick_params(labelsize = 6)
fig.suptitle('Gráfico de Barras Gruas', fontsize = 10, fontweight = "bold")
plt.show()

# Medidas descriptivas
print(prueba['DockCode'].describe())
