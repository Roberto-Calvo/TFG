#-------------------LIBRERÍAS-----------------

import pandas as pd
from dateutil.relativedelta import relativedelta

#----------------------------------------------

#-------------------FUNCIONES------------------

def rango_fechas(desde, hasta):
    return [desde + relativedelta(days=days) for days in range((hasta - desde).days + 1)]

def convert_to_hours(delta):
    total_seconds = delta.total_seconds()
    hours = str(int(total_seconds // 3600)).zfill(2)
    minutes = str(int((total_seconds % 3600) // 60)).zfill(2)
    seconds = str(int(total_seconds % 60)).zfill(2)
    return f"{hours}:{minutes}:{seconds}"

#----------------------------------------------

#-------------------MAIN-----------------------

# Seleccion de variables
fechas_amura = [
    'Hora_Inicio_Servicio_Entrada', 'Hora_Fin_Servicio_Entrada',
    'Hora_Inicio_Servicio_Salida', 'Hora_Fin_Servicio_Salida',
    'Entrada_AP2', 'Entrada_AP1', 'Salida_AP1', 'Salida_AP2'
    ]

variables_amura = [
    'IMO', 'Escala', 'Hora_Inicio_Servicio_Entrada', 'Hora_Fin_Servicio_Entrada',
    'Hora_Inicio_Servicio_Salida', 'Hora_Fin_Servicio_Salida', 'Muelle', 'GT',
    'Eslora', 'Manga', 'Entrada_AP2', 'Entrada_AP1', 'Salida_AP1', 'Salida_AP2',
    'Latitud_Amarre', 'Longitud_Amarre', 'Latitud_Amarre_2', 'Longitud_Amarre_2',
    'Calado_Popa_Entrada','Calado_Popa_Salida'
]
coordenadas_amura = ['Latitud_Amarre', 'Longitud_Amarre', 'Latitud_Amarre_2', 'Longitud_Amarre_2']

# Lectura de GRUAS en MDES - DataSet.xslx
variables_gruas = ['code', 'DockCode']
gruas = pd.read_excel('MDES - DataSet.xlsx', sheet_name='Grúas', na_values=['NULL',''])
gruas.rename(columns={'Number':'code'}, inplace=True)
gruas_filtrado = gruas[variables_gruas]

# Lectura de MUELLES en MDES - DataSet.xslx
variables_muelles = ['DockCode', 'Muelle']
muelles = pd.read_excel('MDES - DataSet.xlsx', sheet_name='Muelles', na_values=['NULL',''])
muelles.rename(columns={'Name': 'Muelle'}, inplace=True)
muelles['Muelle'].replace('TR.Costa','TRANS. COSTA', inplace=True)
muelles['Muelle'].replace('MUELLE COSTA','COSTA', inplace=True)
muelles['Muelle'].replace('MUELLE DEL ESTE','MUELLE ESTE', inplace=True)
muelles_filtrado = muelles[variables_muelles]


# Lee excel llamado Amura.xslx con el formato adecuado
amura = pd.read_excel('Amura.xlsx', parse_dates=fechas_amura, date_format="%Y-%m-%d %H:%M:%S.%f",na_values=['NULL',''] )
amura_filtrado = amura[variables_amura]

# Tratamiento de los missing values
filas_nan = [
    'Escala', 'Hora_Inicio_Servicio_Entrada', 'Hora_Fin_Servicio_Entrada',
    'Hora_Fin_Servicio_Salida', 'Muelle', 'GT',
    'Entrada_AP2', 'Entrada_AP1', 'Salida_AP1', 'Salida_AP2',
    'Latitud_Amarre', 'Longitud_Amarre', 'Latitud_Amarre_2', 'Longitud_Amarre_2',
]
amura_filtrado=amura_filtrado.dropna(subset=filas_nan)

# Une amura con muelles en la tabla union_final
union_final = amura_filtrado.merge(muelles_filtrado, on='Muelle')
union_final['DockCode'] = union_final['DockCode'].astype('int64')

# Columnas de inicio y final de los dias transcurridos, y tiempo transcurrido
union_final["inicio"] = union_final["Hora_Inicio_Servicio_Entrada"].dt.date
union_final["fin"] = union_final["Hora_Fin_Servicio_Salida"].dt.date
union_final['tiempo'] = union_final['Hora_Fin_Servicio_Salida']-union_final['Hora_Inicio_Servicio_Entrada']

# Lectura de archivos cranes por cada dia que pasa el barco en el muelle
crane_status_cache = {} 
margen = 0.00025 # Margen que se le suma a las latitudes y longitudes (aproximadamente 30 metros)
# Busca por filas las posibles gruas utilizadas en cada operacion, la información diaria de las 
# gruas se halla en los archivos 'Cranes/(fecha).csv'
for index, row in union_final.iterrows():
    i=0
    inicio = row['inicio']
    fin = row['fin']

    # Acota latitud y longitud en un rectángulo para aceptar las grúas que pudieron participar en dicha operación, 
    # además le resta y suma el margen, dando la posibilidad de que las grúas se puedan ubicar más lejos que los amarres
    # Latitudes minima y maxima
    min_lat = min(row['Latitud_Amarre'],row['Latitud_Amarre_2'])-margen
    max_lat = max(row['Latitud_Amarre'],row['Latitud_Amarre_2'])+margen
    # Longitudes minima y maxima (son negativas)
    min_lon = min(row['Longitud_Amarre'],row['Longitud_Amarre_2'])+margen
    max_lon = max(row['Longitud_Amarre'],row['Longitud_Amarre_2'])-margen

    if pd.isna(inicio) or pd.isna(fin):
        continue # Salta la fila si la fecha es NaN
    dias = rango_fechas(inicio,fin) # Total de días transcurridos
    union_final.loc[index,'dias'] = len(dias)
    nombres_archivos = [f'Cranes/{fecha.strftime("%Y-%m-%d")}.csv' for fecha in dias]

    for archivo in nombres_archivos:
        if archivo not in crane_status_cache:
            try:
                crane_status_cache[archivo] = pd.read_csv(archivo) 
                crane_status_cache[archivo] = crane_status_cache[archivo].merge(gruas_filtrado, on='code')
            except FileNotFoundError:
                print(f'El archivo {archivo} no se encontró.')
                continue # Salta la fila si no se encuentra el csv en Cranes para ese dia
        cranes_csv = crane_status_cache[archivo]
        dock_code = row['DockCode']

        # Filtra y guarda en cache las gruas operativas el dia de la operacion
        filtered_cranes = cranes_csv[(cranes_csv['status'] == 1) & (cranes_csv['boom_angle'] == 0)]

        # Obtiene las gruas relevantes ese dia, de acuerdo al muelle que llega el barco correspondiente a la operacion
        # y al rectángulo de aceptación anteriormente definido. Luego, coloca ese número de grúas del día j, en la columna diaj'
        relevant_cranes = filtered_cranes[(filtered_cranes['DockCode'] == dock_code)
            & (filtered_cranes['lon']>=min_lon) & (filtered_cranes['lon']<=max_lon)
            & (filtered_cranes['lat']>=min_lat) & (filtered_cranes['lat']<=max_lat)                                                    
                                                                ]
        union_final.loc[index,f'dia{i}'] = len(relevant_cranes)
        i=i+1

#Lista con las columnas de cada dia
crane_columns = [f'dia{num}' for num in range(100)]

# Calcula la media aritmetica de las gruas que estuvieron operativas los dias que paso el barco en el muelle , y coloca las grúas 
# disponibles el primer dia en la columna 'gruas'      
union_final['gruas_mean'] = (union_final[crane_columns].sum(axis=1,min_count=1))/union_final['dias']
union_final['gruas'] = union_final['dia0']

# Pasa a excel el dataset con las variables deseadas
lista = ['Hora_Inicio_Servicio_Entrada','Hora_Fin_Servicio_Salida','dias','tiempo',
          'gruas','gruas_mean','DockCode', 'GT', 'Eslora', 'Manga','Calado_Popa_Entrada','Calado_Popa_Salida'
        ]

union_final['tiempo'] = union_final['tiempo'].apply(convert_to_hours)

excel = union_final[lista]
prueba = excel[excel['gruas_mean']>0]

# Escribe archivo excel 'prueba.xlsx'
with pd.ExcelWriter('prueba.xlsx') as writer:
    prueba.to_excel(writer)
    writer.close()
