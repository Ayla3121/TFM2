#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import lightgbm as lgb  
import joblib  
import lightgbm as lgb
import matplotlib
import sklearn


# In[2]:


def information(File):
    num_values = File.count()  # Número de valores no nulos
    unicos = File.nunique()    # Número de valores únicos
    num_nan = File.isna().sum()  # Número de valores NaN

    info_df = pd.DataFrame({
        'Column': File.columns,
        'Non-null Count': num_values.values,
        'DataType': File.dtypes.values,
        'Nunique': unicos.values,
        'NaN Count': num_nan.values  # Añadir la columna con el número de NaN
    })
    
    return info_df


# In[ ]:


# Título de la app
st.title('Predicción de Tarifas de Transporte')

# Subir archivo
uploaded_file_envios = st.file_uploader("Sube el archivo .csv o .xlsx en el que deseas hacer predicciones", type=['xlsx', 'csv'])

if uploaded_file_envios is not None:
    # Leer el archivo subido
    if uploaded_file_envios.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file_envios)
    elif uploaded_file_envios.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file_envios)
       
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    st.success("Archivo cargado correctamente.")


# In[ ]:


# Subir archivo del modelo (joblib o pickle)
uploaded_model = st.file_uploader("Sube el archivo del modelo (.pkl o .joblib)", type=['pkl', 'joblib'])

# Cargar el archivo del modelo
if uploaded_model is not None:
    try:
        model_lightgbm_HAND_red2 = joblib.load(uploaded_model)
        # Asegurarse de que el modelo tiene el método 'predict'
        if not hasattr(model_lightgbm_HAND_red2, 'predict'):
            raise ValueError("El modelo cargado no tiene el método 'predict'. Verifica el archivo del modelo.")
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"Error al cargar el archivo del modelo: {e}")


# In[ ]:


# Cargar el archivo 'historico_envios' subido por el usuario
uploaded_file_envios2 = st.file_uploader("Sube el archivo 'historico_envios'", type=['xlsx', 'csv'])

if uploaded_file_envios2 is not None:
    # Leer el archivo subido
    if uploaded_file_envios2.name.endswith('.xlsx'):
        historico_envios = pd.read_excel(uploaded_file_envios2)
    elif uploaded_file_envios2.name.endswith('.csv'):
        historico_envios = pd.read_csv(uploaded_file_envios2)
    st.success("Archivo 'historico_envios' cargado correctamente.")


# In[ ]:


# Cargar el archivo 'historico_tarifas' subido por el usuario
uploaded_file_envios3 = st.file_uploader("Sube el archivo 'historico_tarifas'", type=['xlsx', 'csv'])

if uploaded_file_envios3 is not None:
    # Leer el archivo subido
    if uploaded_file_envios3.name.endswith('.xlsx'):
        historico_tarifas = pd.read_excel(uploaded_file_envios3)
    elif uploaded_file_envios3.name.endswith('.csv'):
        historico_tarifas = pd.read_csv(uploaded_file_envios3)
    st.success("Archivo 'historico_tarifas' cargado correctamente.")


# In[ ]:


# Opción para ver las versiones iniciales
if st.checkbox("Ver los dataframes originales antes del procesamiento"):
    st.write("Dataframe df original:")
    st.dataframe(df.head())
    st.write("Información de df:")
    st.dataframe(information(df))
    
    st.write("Dataframe Historico_Tarifas original:")
    st.dataframe(historico_tarifas.head())
    st.write("Información de historico_tarifas:")
    st.dataframe(information(historico_tarifas))
    
    st.write("Dataframe historico_envios original:")
    st.dataframe(historico_envios.head())
    st.write("Información de historico_envios:")
    st.dataframe(information(historico_envios))


# In[4]:


# Unificación nomenclatura

df.loc[df['numero_pales'] == 0, 'numero_pales'] = 33
historico_tarifas.loc[historico_tarifas['numero_pales'] == 0, 'numero_pales'] = 33

historico_envios = historico_envios.rename(columns={'region_destino_final': 'region_destino','fecha_real_carga':'fecha',
                                       'provincia_destino_final':'provincia_destino'})


# In[5]:


df = df[(df['id_pais_origen'] == 'ES') & 
        (df['id_pais_destino'] == 'ES') & 
        (df['region_origen'] == 30) & 
        (df['incoterms'] == 'DAP') & 
        (df['id_tipo_transporte'].isin([1, 3]))]

historico_tarifas = historico_tarifas[(historico_tarifas['id_pais_origen'] == 'ES') &
        (historico_tarifas['id_pais_destino'] == 'ES') &
        (historico_tarifas['region_origen'] == 30) & 
        (historico_tarifas['incoterms'] == 'DAP') & 
        (historico_tarifas['id_tipo_transporte'].isin([1, 3]))]

historico_envios = historico_envios[(historico_envios['id_pais_origen'] == 'ES') &
        (historico_envios['id_pais_destino'] == 'ES') &
        (historico_envios['region_origen'] == 30) & 
        (historico_envios['incoterms'] == 'DAP') & 
        (historico_envios['id_tipo_transporte'].isin([1, 3]))]


# In[6]:


# Eliminar columnas no necesarias

df = df.drop(columns=['incoterms','id_pais_destino','id_pais_origen'])
historico_tarifas = historico_tarifas.drop(columns=['incoterms','id_pais_destino','id_pais_origen'])
historico_envios = historico_envios.drop(columns=['incoterms','id_pais_destino','id_pais_origen'])


# In[7]:


# Eliminar filas con NaN en 'id_tipo_transporte', 'numero_pales' y 'fecha'
df = df.dropna(subset=['id_tipo_transporte', 'numero_pales', 'fecha'])

historico_tarifas = historico_tarifas.dropna(subset=['id_tipo_transporte', 'numero_pales', 'fecha'])

historico_envios = historico_envios.dropna(subset=['id_tipo_transporte', 'numero_pales', 'fecha'])


# In[8]:


# Función para asignar la campaña
def asignar_campaña(fecha):
    year = fecha.year
    return f'{year}-{year + 1}' if fecha.month >= 9 else f'{year - 1}-{year}'

# Convertir formato de la columna 'fecha' a datetime
df['fecha'] = pd.to_datetime(df['fecha'])
historico_tarifas['fecha'] = pd.to_datetime(historico_tarifas['fecha'])

historico_envios['fecha'] = pd.to_datetime(historico_envios['fecha'])
historico_envios['fecha'] = historico_envios['fecha'].dt.date

# Asignar la campaña
df['campaña'] = df['fecha'].apply(asignar_campaña)
historico_tarifas['campaña'] = historico_tarifas['fecha'].apply(asignar_campaña)
historico_envios['campaña'] = historico_envios['fecha'].apply(asignar_campaña)


# In[9]:


# Ordenar 'Historico_Tarifas' por las columnas indicadas
historico_tarifas = historico_tarifas.sort_values(by=['campaña', 'provincia_origen', 'provincia_destino', 'id_tipo_transporte', 'numero_pales', 'fecha'])

# Agrupar y obtener el precio más reciente
historico_tarifas_agrupado = historico_tarifas.groupby(
    ['campaña', 'provincia_origen', 'provincia_destino', 'id_tipo_transporte', 'numero_pales']
).agg(precio_reciente=('precio', 'last')).reset_index()


# In[10]:


# Realizar el merge para agregar 'precio_reciente' al DataFrame original
historico_tarifas = historico_tarifas.merge(
    historico_tarifas_agrupado,
    on=['campaña', 'provincia_origen', 'provincia_destino', 'id_tipo_transporte', 'numero_pales'],
    how='left'
)


# In[11]:


historico_tarifas=historico_tarifas.drop(columns=['precio'])


# In[ ]:


historico_tarifas['fecha'] = pd.to_datetime(historico_tarifas['fecha'])
historico_tarifas['year'] = historico_tarifas['fecha'].dt.year

historico_envios['fecha'] = pd.to_datetime(historico_envios['fecha'])
historico_envios['year'] = historico_envios['fecha'].dt.year


# In[12]:


# Eliminar la columna 'fecha' y eliminar duplicados (excepto en historico_envios)
df = df.drop(columns=['fecha'])
historico_tarifas = historico_tarifas.drop(columns=['fecha'])
historico_envios = historico_envios.drop(columns=['fecha'])

df = df.drop_duplicates()
historico_tarifas = historico_tarifas.drop_duplicates()


# In[13]:


# Mapear las campañas a números
campaña_mapping = {'2020-2021': 2020, '2021-2022': 2021, '2022-2023': 2022, '2023-2024': 2023, '2024-2025': 2024}

df['campaña_codificada'] = df['campaña'].map(campaña_mapping)

historico_tarifas['campaña_codificada'] = historico_tarifas['campaña'].map(campaña_mapping)

historico_envios['campaña_codificada'] = historico_envios['campaña'].map(campaña_mapping)


# In[14]:


df=df.drop(columns=['campaña'])
historico_tarifas=historico_tarifas.drop(columns=['campaña'])
historico_envios=historico_envios.drop(columns=['campaña'])


# In[15]:


# Convertir a formato entero las columnas necesarias
cols_int = ['region_origen', 'region_destino', 'id_tipo_transporte', 'numero_pales']
df[cols_int] = df[cols_int].astype(int)

historico_tarifas[cols_int] = historico_tarifas[cols_int].astype(int)

cols_int2 = ['region_origen', 'region_destino', 'id_tipo_transporte', 'numero_pales','incidencia']
historico_envios[cols_int2] = historico_envios[cols_int2].astype(int)


# In[16]:


# Diccionario de provincias y regiones
diccionario_provincias = {'Murcia': 30, 'Huelva': 21, 'Alicante': 3, 'Álava': 1, 'Albacete': 2, 
                          'Almería': 4, 'Ávila': 5, 'Badajoz': 6, 'Baleares': 7, 'Barcelona': 8, 'Burgos': 9, 
                          'Cáceres': 10, 'Cádiz': 11, 'Castellón': 12, 'Ciudad Real': 13, 'Córdoba': 14, 
                          'La Coruña': 15, 'Cuenca': 16, 'Gerona': 17, 'Granada': 18, 'Guadalajara': 19, 
                          'Guipúzcoa': 20, 'Huesca': 22, 'Jaén': 23, 'León': 24, 'Lérida': 25, 'La Rioja': 26, 
                          'Lugo': 27, 'Madrid': 28, 'Málaga': 29, 'Navarra': 31, 'Orense': 32, 'Asturias': 33, 
                          'Palencia': 34, 'Las Palmas': 35, 'Pontevedra': 36, 'Salamanca': 37,
                          'S.C de Tenerife':38,'Cantabria': 39, 
                          'Segovia': 40, 'Sevilla': 41, 'Soria': 42, 'Tarragona': 43, 'Teruel': 44, 'Toledo': 45, 
                          'Valencia': 46, 'Valladolid': 47, 'Vizcaya': 48, 'Zamora': 49, 'Zaragoza': 50, 'Ceuta': 51,'Melilla':52}


# In[17]:


# Filtrar regiones destino no válidas
df = df[~df['region_destino'].isin([7,35,51,52,38])]

# Filtrar regiones destino no válidas
historico_tarifas = historico_tarifas[~historico_tarifas['region_destino'].isin([7,35,38,51,52])]

# Filtrar regiones destino no válidas
historico_envios = historico_envios[~historico_envios['region_destino'].isin([7,35,38,51,52])]


# In[18]:


# Eliminar filas si ambos 'region_destino' y 'provincia_destino' son NaN
df = df[~(df['region_destino'].isna() & df['provincia_destino'].isna())]

historico_tarifas = historico_tarifas[~(historico_tarifas['region_destino'].isna() & historico_tarifas['provincia_destino'].isna())]

historico_envios = historico_envios[~(historico_envios['region_destino'].isna() & historico_envios['provincia_destino'].isna())]


# In[19]:


# Asignar 'region_destino' solo si está vacío, basándose en 'provincia_destino'
df['region_destino'] = df.apply(
    lambda row: diccionario_provincias.get(row['provincia_destino'], row['region_destino']) if pd.isna(row['region_destino']) else row['region_destino'],
    axis=1
)

historico_tarifas['region_destino'] = historico_tarifas.apply(
    lambda row: diccionario_provincias.get(row['provincia_destino'], row['region_destino']) if pd.isna(row['region_destino']) else row['region_destino'],
    axis=1
)

historico_envios['region_destino'] = historico_envios.apply(
    lambda row: diccionario_provincias.get(row['provincia_destino'], row['region_destino']) if pd.isna(row['region_destino']) else row['region_destino'],
    axis=1
)


# In[20]:


# Invertir el diccionario: claves serán las regiones y valores las provincias
diccionario_regiones = {v: k for k, v in diccionario_provincias.items()}

# Asignar 'provincia_destino' si está vacío, basándose en 'region_destino'
df['provincia_destino'] = df.apply(
    lambda row: diccionario_regiones.get(row['region_destino'], row['provincia_destino']) if pd.isna(row['provincia_destino']) else row['provincia_destino'],
    axis=1
)

historico_tarifas['provincia_destino'] = historico_tarifas.apply(
    lambda row: diccionario_regiones.get(row['region_destino'], row['provincia_destino']) if pd.isna(row['provincia_destino']) else row['provincia_destino'],
    axis=1
)

historico_envios['provincia_destino'] = historico_envios.apply(
    lambda row: diccionario_regiones.get(row['region_destino'], row['provincia_destino']) if pd.isna(row['provincia_destino']) else row['provincia_destino'],
    axis=1
)



# In[21]:


# Eliminar registros que tengan algún campo vacío
df = df.dropna()
historico_tarifas = historico_tarifas.dropna()
historico_envios = historico_envios.dropna()


# In[22]:


# Función para obtener la comunidad autónoma según la provincia
def obtener_comunidad(provincia, comunidades):
    for comunidad, provincias in comunidades.items():
        if provincia in provincias:
            return comunidad
    return None


comunidades_autonomas = {
    "Andalucía": ["Almería", "Cádiz", "Córdoba", "Granada", "Huelva", "Jaén", "Málaga", "Sevilla"],
    "Aragón": ["Huesca", "Teruel", "Zaragoza"],
    "Asturias": ["Asturias"],
    "Islas Baleares": ["Baleares"],
    "Canarias": ["Las Palmas", "Santa Cruz de Tenerife"],
    "Cantabria": ["Cantabria"],
    "Castilla-La Mancha": ["Albacete", "Ciudad Real", "Cuenca", "Guadalajara", "Toledo"],
    "Castilla y León": ["Ávila", "Burgos", "León", "Palencia", "Salamanca", "Segovia", "Soria", "Valladolid", "Zamora"],
    "Cataluña": ["Barcelona", "Gerona", "Lérida", "Tarragona"],
    "Extremadura": ["Badajoz", "Cáceres"],
    "Galicia": ["La Coruña", "Lugo", "Orense", "Pontevedra"],
    "Madrid": ["Madrid"],
    "Murcia": ["Murcia"],
    "Navarra": ["Navarra"],
    "La Rioja": ["La Rioja"],
    "País Vasco": ["Álava", "Guipúzcoa", "Vizcaya"],
    "Comunidad Valenciana": ["Alicante", "Castellón", "Valencia"],
    "Ceuta": ["Ceuta"],
    "Melilla": ["Melilla"]
}





# In[23]:


codigos_comunidades={'Andalucía': 0, 'Aragón': 1, 'Asturias': 2, 'Islas Baleares': 12, 
                     'Canarias': 3, 'Cantabria': 4, 'Castilla-La Mancha': 6, 'Castilla y León': 5, 
                     'Cataluña': 7, 'Extremadura': 10, 'Galicia': 11, 'Madrid': 14, 'Murcia': 16, 
                     'Navarra': 17, 'La Rioja': 13, 'País Vasco': 18, 
                     'Comunidad Valenciana': 9, 'Ceuta': 8, 'Melilla': 15}


# In[24]:


# Crear la columna 'Comunidad_destino_codificada'
df['comunidad_destino'] = df['provincia_destino'].apply(lambda x: obtener_comunidad(x, comunidades_autonomas))
historico_tarifas['comunidad_destino'] = historico_tarifas['provincia_destino'].apply(lambda x: obtener_comunidad(x, comunidades_autonomas))
historico_envios['comunidad_destino'] = historico_envios['provincia_destino'].apply(lambda x: obtener_comunidad(x, comunidades_autonomas))

# Codificar la columna de comunidades autónomas
df['comunidad_destino_codificada'] = df['comunidad_destino'].map(codigos_comunidades)
historico_tarifas['comunidad_destino_codificada'] = historico_tarifas['comunidad_destino'].map(codigos_comunidades)
historico_envios['comunidad_destino_codificada'] = historico_envios['comunidad_destino'].map(codigos_comunidades)


# In[25]:


diccionario_km={1: 803, 2: 163, 3: 80, 4: 221, 5: 509, 6: 731, 7: 206, 8: 595, 9: 739, 10: 633, 
                11: 584, 12: 320, 13: 327, 14: 434, 15: 1074, 16: 366, 17: 667, 18: 362, 19: 456, 
                20: 793, 21: 656, 22: 655, 23: 339, 24: 753, 25: 686, 26: 727, 27: 904, 28: 404, 29: 423, 
                30: 0, 31: 749, 32: 901, 33: 926, 34: 632, 35: 1938, 36: 1087, 37: 614, 38: 2037, 39: 800, 
                40: 491, 41: 579, 42: 618, 43: 525, 44: 412, 45: 389, 46: 239, 
                47: 612, 48: 795, 49: 655, 50: 688, 51: 640, 52:511}

# Crear columna 'km' utilizando el diccionario_km basado en 'region_destino'
df['km'] = df['region_destino'].map(diccionario_km)

# Verificar si hay valores nulos en la columna 'km' y reportar si es necesario
if df['km'].isnull().any():
    st.warning('Algunas regiones destino no tienen kilómetros asignados en el diccionario.')


# In[26]:


diccionario_cota={'Ávila': 1131, 'Soria': 1093, 'León': 1080, 'Granada': 1077, 'Guadalajara': 1068, 'Teruel': 1055, 
                  'Segovia': 1020, 'Cuenca': 958, 'Palencia': 940, 'Lérida': 932, 'Burgos': 914, 'La Rioja': 860, 
                  'Huesca': 850, 'Albacete': 845, 'Zamora': 828, 'Salamanca': 823, 'Madrid': 817, 
                  'S.C. de Tenerife': 809, 'Valladolid': 777, 'Orense': 761, 'Almería': 727, 'Jaén': 714, 
                  'Ciudad Real': 712, 'Álava': 663, 'Toledo': 635, 'Asturias': 626, 'Cantabria': 603, 
                  'Navarra': 601, 'Castellón': 593, 'Lugo': 563, 'Zaragoza': 551, 'Málaga': 517, 
                  'Barcelona': 513, 'Gerona': 511, 'Valencia': 507, 'Murcia': 502, 'Cáceres': 455, 
                  'Córdoba': 449, 'Badajoz': 406, 'Alicante': 398, 'Guipúzcoa': 391, 'Pontevedra': 351, 
                  'Tarragona': 339, 'Las Palmas': 307, 'Vizcaya': 289, 'La Coruña': 276, 'Huelva': 222, 
                  'Sevilla': 200, 'Cádiz': 186, 'Baleares': 138, 'Ceuta': 95, 'Melilla': 40}


# Crear las columnas de cota_origen y cota_destino usando el diccionario_cota
df['cota_origen'] = df['provincia_origen'].map(diccionario_cota)
df['cota_destino'] = df['provincia_destino'].map(diccionario_cota)

# Calcular la diferencia de cotas
df['diferencia_cota'] = df['cota_destino'] - df['cota_origen']

# Verificar si hay valores nulos en las columnas de cota
if df[['cota_origen', 'cota_destino']].isnull().any().any():
    st.warning('Algunas provincias no tienen cotas asignadas en el diccionario.')

# Eliminar las columnas temporales si ya no son necesarias
df = df.drop(columns=['cota_origen', 'cota_destino'])



# In[ ]:


historico_tarifas.head()


# In[ ]:


information(historico_tarifas)


# In[27]:


# Crear la columna 'precioX33' en función de 'precio_reciente' y 'numero_pales'
historico_tarifas['precioX33'] = historico_tarifas.apply(
    lambda row: row['precio_reciente'] * 33 if row['numero_pales'] != 33 else row['precio_reciente'], axis=1
)


# In[28]:


historico_tarifas.head()


# In[29]:


# Crear una columna en 'df' con la campaña anterior
df['campaña_anterior'] = df['campaña_codificada'] - 1

# Realizar el merge para agregar 'precio_anterior'
df = df.merge(
    historico_tarifas[['campaña_codificada', 'provincia_origen', 'provincia_destino', 'id_tipo_transporte', 'numero_pales', 'precioX33']],
    left_on=['campaña_anterior', 'provincia_origen', 'provincia_destino', 'id_tipo_transporte', 'numero_pales'],
    right_on=['campaña_codificada', 'provincia_origen', 'provincia_destino', 'id_tipo_transporte', 'numero_pales'],
    how='left',
    suffixes=('', '_precio_anterior')
)


# In[32]:


df.head()


# In[31]:


# Renombrar columna 'precioX33' a 'precio_anterior'
df = df.rename(columns={'precioX33': 'precio_anterior'})

# Eliminar columna temporal 'campaña_anterior'
df = df.drop(columns=['campaña_anterior', 'campaña_codificada_precio_anterior'])


# In[ ]:


# Agrupar por 'campaña_codificada', 'region_origen', 'region_destino', 'id_tipo_transporte' y calcular la suma de 'incidencia'
historico_envios_agrupado = historico_envios.groupby(
    ['year', 'region_origen', 'region_destino', 'id_tipo_transporte']
)['incidencia'].sum().reset_index()


# In[ ]:


# Renombrar la columna calculada a 'pales_campaña'
historico_envios_agrupado.rename(columns={'incidencia': 'incidencia_year'}, inplace=True)


# In[34]:


# Agrupar por 'campaña_codificada', 'region_origen', 'region_destino', 'id_tipo_transporte' y calcular la suma de 'numero_pales'
historico_envios_agrupado2 = historico_envios.groupby(
    ['year', 'region_origen', 'region_destino', 'id_tipo_transporte']
)['numero_pales'].sum().reset_index()


# In[36]:


# Renombrar la columna calculada a 'pales_campaña'
historico_envios_agrupado2.rename(columns={'numero_pales': 'pales_year'}, inplace=True)


# In[ ]:


# Añadimos la columna 'año_anterior' en df
df['año_anterior'] = df['campaña_codificada'] - 1

# Cambiamos el nombre de las columnas 
historico_envios_agrupado.rename(columns={'year': 'año_anterior', 'incidencia_year': 'incidencia_anterior'}, inplace=True)

# Merge para añadir la columna 'incidencia_anterior'
df = df.merge(historico_envios_agrupado, on=['region_origen', 'region_destino', 'id_tipo_transporte', 'año_anterior'], how='left')


# In[ ]:


# Añadimos la columna 'año_anterior' en TM
df['año_anterior'] = df['campaña_codificada'] - 1

# Cambiamos el nombre de las columnas en df
historico_envios_agrupado2.rename(columns={'year': 'año_anterior', 'pales_year': 'pales_anterior'}, inplace=True)

# Fusionamos TM con df para añadir la columna 'incidencia_anterior'
df = df.merge(historico_envios_agrupado2, on=['region_origen', 'region_destino', 'id_tipo_transporte', 'año_anterior'], how='left')


# In[42]:


# Rellenar valores nulos en 'precio_anterior', 'pales_anterior', e 'incidencia_anterior' con 0
df['precio_anterior'] = df['precio_anterior'].fillna(0)
df['pales_anterior'] = df['pales_anterior'].fillna(0)
df['incidencia_anterior'] = df['incidencia_anterior'].fillna(0)


# In[45]:


# Ordenar el dataframe según el orden que necesitas
columnas_ordenadas = ['comunidad_destino_codificada', 'km', 'diferencia_cota', 'campaña_codificada', 'pales_anterior', 
                      'incidencia_anterior', 'precio_anterior', 'region_destino', 'id_tipo_transporte', 'numero_pales']

df = df[columnas_ordenadas]


# In[ ]:


if st.checkbox("Ver el df después del procesamiento"):
    st.write("Dataframe df procesado:")
    st.dataframe(df.head())
    st.write("Información de df después del procesamiento:")
    st.dataframe(information(df))


# In[ ]:


if st.checkbox("Realizar las predicciones"):

    try:
        # Realizar predicciones con el modelo entrenado
        df['precioX33_predicho'] = model_lightgbm_HAND_red2.predict(df)

        # Crear la columna 'prediccion_final' aplicando la transformación solicitada
        df['prediccion_final'] = df.apply(lambda row: row['precioX33_predicho'] / 33 if row['numero_pales'] != 33 else row['precioX33_predicho'], axis=1)

        # Eliminar la columna 'precioX33_predicho'
        df = df.drop(columns=['precioX33_predicho'])
        
        st.write("Head del df con predicciones:")

        # Mostrar el DataFrame con las predicciones
        st.write(df.head())  # Muestra las primeras filas del dataframe con predicciones

        # Permitir al usuario descargar el DataFrame con predicciones
        df_predicciones = df.to_csv(index=False)
        st.download_button("Descargar df con predicciones", df_predicciones, "predicciones.csv", "text/csv")

    except Exception as e:
        st.error(f"Ocurrió un error al hacer las predicciones: {e}")



# In[ ]:


# Función para obtener la mejor tarifa
def obtener_mejor_tarifa(df, region_destino, id_tipo_transporte, campaña_codificada, numero_pales_deseados):
    # Filtrar el DataFrame según los criterios del usuario
    df_filtrado = df[
        (df['region_destino'] == region_destino) &
        (df['id_tipo_transporte'] == id_tipo_transporte) &
        (df['campaña_codificada'] == campaña_codificada)
    ]
    
    # Si no se encuentran tarifas, devolver un mensaje
    if df_filtrado.empty:
        return "No se encontraron tarifas para los criterios especificados."

    # Filtrar por el número de palés
    df_filtrado = df_filtrado[
        (df_filtrado['numero_pales'] <= numero_pales_deseados) | 
        (df_filtrado['numero_pales'] == 33)
    ]
    
    # Si después del filtro no hay tarifas válidas, devolver un mensaje
    if df_filtrado.empty:
        return "No hay tarifas ajustadas para los criterios especificados."

    # Calcular el precio ajustado
    df_filtrado['precio_ajustado'] = df_filtrado.apply(
        lambda row: row['prediccion_final'] if row['numero_pales'] == 33 else row['prediccion_final'] * numero_pales_deseados, axis=1
    )

    # Obtener la fila con la mejor tarifa (menor precio ajustado)
    mejor_tarifa = df_filtrado.loc[df_filtrado['precio_ajustado'].idxmin()]

    # Convertir la Serie a DataFrame
    mejor_tarifa = pd.DataFrame([mejor_tarifa])

    # Seleccionar las columnas necesarias
    mejor_tarifa = mejor_tarifa[['region_destino', 'id_tipo_transporte', 'numero_pales', 'prediccion_final', 'precio_ajustado']]

    # Renombrar las columnas
    mejor_tarifa = mejor_tarifa.rename(columns={'prediccion_final': 'precio/palet', 'precio_ajustado': 'precio total'})
    
    # Redondear los precios a 1 decimal
    mejor_tarifa['precio/palet'] = mejor_tarifa['precio/palet'].round(1)
    mejor_tarifa['precio total'] = mejor_tarifa['precio total'].round(1)
    
    # Devolver el DataFrame con los nuevos nombres
    return mejor_tarifa

# Título para la sección de la calculadora
st.markdown("## Calculadora de mejor precio")

# Inputs del usuario
region_destino = st.selectbox("Elige la región destino", df['region_destino'].unique())

# Desplegable para id_tipo_transporte
id_tipo_transporte = st.selectbox("Elige el tipo de transporte", [1, 3])

# Desplegable para la campaña (solo 2024)
campaña_codificada = st.selectbox("Elige la campaña", [2024])

# Desplegable para número de palés (1 a 33)
numero_pales_deseados = st.selectbox("Número de palés a enviar", range(1, 34))

# Botón para calcular la mejor tarifa
if st.button("Calcular mejor tarifa"):
    mejor_tarifa = obtener_mejor_tarifa(df, region_destino, id_tipo_transporte, campaña_codificada, numero_pales_deseados)
    st.write("La mejor tarifa encontrada es:")
    st.write(mejor_tarifa)





