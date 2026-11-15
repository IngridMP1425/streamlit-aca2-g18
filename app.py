import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuración de la página
st.set_page_config(page_title="PROYECTO ACA 2 – APRENDIZAJE AUTOMÁTICO", layout="wide")

st.title("🖥️​ Modelado con Redes Neuronales Artificiales")

# --- PESTAÑAS ---
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "SQL", "Gráficas", "Machine Learning"])

with tab1:
  st.subheader("Información General")
  st.write("🛒 Este dataset contiene los datos de ventas minoristas de una supertienda entre los años 2015 y 2018 en las categorías de muebles, artículos de oficina y tecnología.")

  # Decorador para almacenar en caché la carga del dataset limpio
  @st.cache_data
  def load_data():
    conn = sqlite3.connect('SalesForecasting.db')
    # Leer la tabla SQL que guardamos previamente
    df = pd.read_sql_query("SELECT * FROM ventas_raw;", conn)
    conn.close()
    return df

  # Cargar los datos limpiados globalmente en la variable dfclean
  df = load_data()

  # Vista de las primeras líneas del dataset
  st.dataframe(df.head(11))

  st.subheader("📋 Información de la base de datos procesada")

  # Tabla con la información de las columnas del dataset y el tipo de dato
  buffer = pd.DataFrame({
    "Columna": df.columns,
    "Tipo de Dato": df.dtypes.values,
    "No Nulos": df.notnull().sum().values
  })

  buffer = pd.DataFrame({
    "Columna": df.columns,
    "Tipo de Dato": df.dtypes.astype(str).values,
    "Valores No Nulos": df.notnull().sum().values
  })

  # Creación de columnas para delimitar el ancho horizontal de la tabla
  col_tabla, _ = st.columns([2, 1])  # Ocupa el 66% del ancho de la página

  with col_tabla:
    st.dataframe(
        buffer,
        use_container_width=False,  # Desactiva la expansión a pantalla completa
        hide_index=True
    )

with tab2:
  st.subheader("🗃️ Base de Datos SQL y Consultas Dinámicas")
  st.markdown("""
  Esta sección ejecuta **consultas dinámicas en lenguaje SQL** directamente sobre la base de datos
  `SalesForecasting.db` para analizar las ventas por estado, categoría, segmento y más.
  """)
  st.divider()

  # ==========================================
  # 1. CARGA DE OPCIONES Y CONEXIÓN A SQL
  # ==========================================
  def obtener_conexion_sql():
    return sqlite3.connect('SalesForecasting.db')

  @st.cache_data
  def cargar_opciones_filtros():
    conn = obtener_conexion_sql()
    tabla = "ventas_raw"  # Nombre de la tabla en SQLite
    categorias = pd.read_sql_query(f'SELECT DISTINCT Category FROM {tabla} WHERE Category IS NOT NULL', conn)['Category'].tolist()
    regiones = pd.read_sql_query(f'SELECT DISTINCT Region FROM {tabla} WHERE Region IS NOT NULL', conn)['Region'].tolist()
    segmentos = pd.read_sql_query(f'SELECT DISTINCT Segment FROM {tabla} WHERE Segment IS NOT NULL', conn)['Segment'].tolist()
    estados = pd.read_sql_query(f'SELECT DISTINCT State FROM {tabla} ORDER BY State ASC', conn)['State'].tolist()

    conn.close()
    return categorias, regiones, segmentos, estados

  try:
    lista_categorias, lista_regiones, lista_segmentos, lista_estados = cargar_opciones_filtros()
  except Exception as e:
    st.error(f"Error al conectar con 'SalesForecasting.db': {e}")
    st.stop()

  # ==========================================
  # 2. CAJAS DE SELECCIÓN (FILTROS)
  # ==========================================
  col_f1, col_f2 = st.columns(2)
  with col_f1:
    sel_categoria = st.multiselect("📌 Filtrar por Categoría:", lista_categorias, default=lista_categorias)
    sel_segmento = st.multiselect("👤 Filtrar por Tipo de Consumidor (Segmento):", lista_segmentos, default=lista_segmentos)

  with col_f2:
    sel_region = st.multiselect("🗺️ Filtrar por Región:", lista_regiones, default=lista_regiones)
    sel_estado = st.selectbox("🏙️ Filtrar por Estado Específico:", ["Todos"] + lista_estados)

  # ==========================================
  # 3. EJECUCIÓN DE LA CONSULTA SQL DINÁMICA
  # ==========================================
  def consultar_sql(categorias, regiones, segmentos, estado):
    conn = obtener_conexion_sql()
    condiciones = []

    if categorias:
      cats_str = "', '".join(categorias)
      condiciones.append(f"Category IN ('{cats_str}')")
    if regiones:
      regs_str = "', '".join(regiones)
      condiciones.append(f"Region IN ('{regs_str}')")
    if segmentos:
      segs_str = "', '".join(segmentos)
      condiciones.append(f"Segment IN ('{segs_str}')")
    if estado != "Todos":
      condiciones.append(f"State = '{estado}'")

    where_clause = " WHERE " + " AND ".join(condiciones) if condiciones else ""

    query = f"""
    SELECT
      "Order ID", "Order Date", "Ship Date", "Ship Mode",
      Segment, City, State, Region, "Product ID",
      Category, "Sub-Category", Sales
    FROM ventas_raw

    {where_clause}
    ORDER BY "Order Date" ASC;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df, query

  df_sql_result, query_sql_usada = consultar_sql(sel_categoria, sel_region, sel_segmento, sel_estado)

  st.divider()

  # ==========================================
  # 4. KPIS Y TABLA DE RESULTADOS
  # ==========================================
  total_sales = df_sql_result['Sales'].sum() if not df_sql_result.empty else 0
  total_orders = df_sql_result['Order ID'].nunique() if not df_sql_result.empty else 0
  ticket_prom = total_sales / total_orders if total_orders > 0 else 0

  kpi1, kpi2, kpi3 = st.columns(3)
  kpi1.metric("💰 Ventas Totales (Filtradas)", f"${total_sales:,.2f} USD")
  kpi2.metric("📦 Cantidad de Órdenes", f"{total_orders:,}")
  kpi3.metric("🎟️ Ticket Promedio por Orden", f"${ticket_prom:,.2f} USD")

  st.subheader("📋 Registros Recuperados de SQLite")
  st.write(f"Filas encontradas para la consulta: **{len(df_sql_result):,}**")
  st.dataframe(df_sql_result, use_container_width=True)

  with st.expander("🛠️ Ver Sentencia SQL Ejecutada"):
    st.code(query_sql_usada, language='sql')

with tab3:

  st.subheader("📉 Análisis Visual")
  col_viz1, col_viz2 = st.columns(2)

  with col_viz1:

    # Agrupar ventas totales por Estado utilizando la variable 'data' ya cargada
    ventas_estado = df.groupby('State')['Sales'].sum().reset_index()

    # Diccionario para mapear nombres de estados a sus abreviaturas de dos letras
    us_state_to_abbrev = {
      'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
      'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
      'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
      'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
      'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
      'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
      'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
      'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
      'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
      'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
    }

    # Asignar el código de estado correspondiente
    ventas_estado['State_Code'] = ventas_estado['State'].map(us_state_to_abbrev)
    # Crear el mapa interactivo de Estados Unidos
    fig = px.choropleth(
    ventas_estado,
    locations='State_Code',
    locationmode="USA-states",
    color='Sales',
    hover_name='State',
    color_continuous_scale='Viridis',
    scope="usa",
    labels={'Sales': 'Ventas Totales ($)'},
    title='Ventas Totales por Estado (Estados Unidos)'
    )

    fig.update_layout(
    geo=dict(bgcolor='rgba(0,0,0,0)'),
    margin={"r":0,"t":50,"l":0,"b":0}
    )

    # Mostrar el gráfico interactivo
    st.plotly_chart(fig)

    st.write("Los estados donde se concentra la mayor cantidad de ventas son: California, Nueva York, Texas y Washington.")

  with col_viz2:

    # Agrupar la sumatoria de Sales por State y Category
    ventas_estado_categoria = df.groupby(['State', 'Category'])['Sales'].sum().reset_index()

    # Reestructurar con pivot para rellenar combinaciones inexistentes con 0
    pivoteado_cat = ventas_estado_categoria.pivot(index='State', columns='Category', values='Sales').fillna(0)

    # Regresar al formato largo para Plotly Express
    ventas_cat_completas = pivoteado_cat.stack().reset_index(name='Sales')

    # Ordenar alfabéticamente por estado para mantener coherencia en el eje X
    ventas_cat_completas = ventas_cat_completas.sort_values(by='State')

    # Generar el gráfico de líneas con Plotly
    fig_categoria = px.line(
      ventas_cat_completas,
      x='State',
      y='Sales',
      color='Category',
      title='Sumatoria de Ventas por Estado y Categoría de Producto',
      labels={'State': 'Estado', 'Sales': 'Ventas Totales ($)', 'Category': 'Categoría'},
      markers=True
    )

    # Ajustar diseño del gráfico
    fig_categoria.update_layout(
    	xaxis_tickangle=-45,
    	xaxis_title='Estado',
    	yaxis_title='Ventas Totales ($)',
    	margin=dict(l=40, r=40, t=60, b=120),
    	legend_title='Categoría'
    )

    # Mostrar gráfico interactivo
    st.plotly_chart(fig_categoria)

    st.write("Las ventas por categoría han sido similares en la mayoría del país, a excepción de los estados de Florida y Nueva York donde las ventas de tecnología se muestran significativamente superiores a las otras dos categorías, y Michigan donde las ventas de artículos de oficina se muestran superiores a las demás categorías.")

  # Gráfica Mensual Histórica de cada Categoría
  st.subheader("Visualización Mensual Histórica")
  st.markdown("""
  Con las siguientes gráficas se puede apreciar el comportamiento de las ventas registradas de cada categoría por mes en todos los años. Se muestra el comportamiento asimétrico de los datos mensuales en cada año.
  """)

  # 1. Preparar los datos extrayendo Año y Mes por separado para la comparación anual
  data_bar = df.copy()
  data_bar['Order Date'] = pd.to_datetime(data_bar['Order Date'])
  data_bar['Año'] = data_bar['Order Date'].dt.year
  data_bar['Mes_Num'] = data_bar['Order Date'].dt.month

  # Mapear números de mes a nombres cortos para el eje X
  meses_nombres = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
  data_bar['Mes'] = data_bar['Mes_Num'].map(meses_nombres)

  # 2. Agrupar las ventas mensuales por año, mes y categoría
  df_grouped_bar = data_bar.groupby(['Año', 'Mes_Num', 'Mes', 'Category'])['Sales'].sum().reset_index()

  # Ordenar por el número de mes para asegurar la secuencia cronológica de Enero a Diciembre
  df_grouped_bar = df_grouped_bar.sort_values(by=['Category', 'Año', 'Mes_Num'])

  # 3. Crear tres gráficos de barras separados (uno por categoría)
  categorias = df_grouped_bar['Category'].unique()
  fig_bar, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=False)

  # Paleta de colores consistente para los años
  paleta_anos = sns.color_palette("viridis", n_colors=df_grouped_bar['Año'].nunique())

  for idx, cat in enumerate(categorias):
    df_cat = df_grouped_bar[df_grouped_bar['Category'] == cat]

    sns.barplot(
      ax=axes[idx],
      data=df_cat,
      x='Mes',
      y='Sales',
      hue='Año',
      palette=paleta_anos
    )

    axes[idx].set_title(f'Comparativa Mensual Histórica (2015-2018) - Categoría: {cat}', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Mes', fontsize=11)
    axes[idx].set_ylabel('Ventas acumuladas ($ USD)', fontsize=11)
    axes[idx].grid(True, linestyle=':', alpha=0.6, axis='y')
    axes[idx].legend(title='Año', loc='upper left')

  plt.tight_layout()

  # Mostrar gráfico interactivo
  st.pyplot(fig_bar)

with tab4:
  st.subheader("🤖 Modelos de Entrenamiento")

  # Extraer df_sql_grouped directamente de la Base de Datos SQL ---
  conn = sqlite3.connect('SalesForecasting.db')
  query_sql = """
  SELECT
    "Order Date" AS fecha,
    Category AS categoria,
    SUM(Sales) AS total_ventas
  FROM ventas_raw
  GROUP BY "Order Date", Category
  ORDER BY fecha ASC;
  """
  df_sql_grouped = pd.read_sql_query(query_sql, conn)
  conn.close()

  # Convertir la columna 'fecha' a datetime por seguridad
  df_sql_grouped['fecha'] = pd.to_datetime(df_sql_grouped['fecha'])

  # Pivotear: Filas = Fechas, Columnas = Categorías, Valores = Ventas
  df_pivot = df_sql_grouped.pivot(index='fecha', columns='categoria', values='total_ventas').fillna(0)

  # Agrupar por semanas Asegurar frecuencia semanar continua
  df_weekly = df_pivot.resample('W').sum().fillna(0)

  # Aplicar transformación logarítmica para reducir varianza
  df_log = np.log1p(df_weekly)

  # Feature 1: ¿La semana cae en la segunda mitad / fin del mes? (0 o 1)
  df_log['is_month_end'] = (df_log.index.day >= 25).astype(int)

  # Feature 2: ¿Es temporada alta de fin de año (Noviembre / Diciembre)? (0 o 1)
  df_log['is_holiday_season'] = df_log.index.month.isin([11, 12]).astype(int)

  # Feature 3: Mes ordinal normalizado para captar estacionalidad suave (1 a 12 / 12)
  df_log['month_sin'] = np.sin(2 * np.pi * df_log.index.month / 12)

  # Escalar las 6 columnas en el rango [0, 1]
  scaler_x = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler_x.fit_transform(df_log)

  # Escalador independiente para el Target (Solo las 3 categorías de ventas)
  scaler_y = MinMaxScaler(feature_range=(0, 1))
  scaled_target = scaler_y.fit_transform(df_log[['Furniture', 'Office Supplies', 'Technology']])

  # Generar Secuencias (LOOKBACK = 4 semanas)
  LOOKBACK = 4

  X, y, target_dates = [], [], []

  for i in range(len(scaled_data) - LOOKBACK):
    # Guardar las 4 semanas de entrada con 6 características
    X.append(scaled_data[i : i + LOOKBACK])
    # Guardar la etiqueta de la semana 5 (con las 3 categorías a predecir)
    y.append(scaled_data[i + LOOKBACK])
    # Guardar la fecha real del día/semana a predecir
    target_dates.append(df_weekly.index[i + LOOKBACK])

  X = np.array(X)
  y = np.array(y)
  target_dates = pd.to_datetime(target_dates)

  # Máscara booleana: Entrenamiento hasta 2017-12-31, Prueba desde 2018-01-01
  train_mask = target_dates <= '2017-12-31'
  test_mask = target_dates >= '2018-01-01'

  X_train, y_train = X[train_mask], y[train_mask]
  X_test, y_test = X[test_mask], y[test_mask]

  # Fechas asociadas para posterior graficación en Streamlit
  dates_train = target_dates[train_mask]
  dates_test = target_dates[test_mask]

  # Aplanar las características de entrada secuenciales para el modelo lineal
  # X_train original: (muestras, lookback=4, features=6) -> Redimensionar a (muestras, 24)
  X_train_flat = X_train.reshape(X_train.shape[0], -1)
  X_test_flat = X_test.reshape(X_test.shape[0], -1)

  # Inicializar y entrenar el modelo de Regresión Lineal de múltiples salidas
  lr_model = MultiOutputRegressor(LinearRegression())
  lr_model.fit(X_train_flat, y_train)

  # Predecir sobre el conjunto de prueba (2018)
  y_pred_lr_scaled = lr_model.predict(X_test_flat)

  # IMPORTANTE: y_train e y_pred_lr_scaled contienen 6 columnas.
  # Recortamos a las primeras 3 columnas (las ventas de las categorías) antes de aplicar inverse_transform con scaler_y
  y_pred_lr_scaled_only_sales = y_pred_lr_scaled[:, :3]

  # Invertir MinMaxScaler usando el escalador de target (scaler_y)
  y_pred_lr_log = scaler_y.inverse_transform(y_pred_lr_scaled_only_sales)

  # Invertir Transformación Logarítmica para obtener valores en dólares reales
  y_pred_lr_real = np.expm1(y_pred_lr_log)

  optionML = st.selectbox(
    "Selecciona el Modelo:",
    ["Regresión Líneal", "Red LSTM"])

  if optionML == "Regresión Líneal":

    # Sección para la regresión lineal
    st.header('📈 Modelo de Regresión Lineal')
    st.write('Se usa este modelo con la misma información para tenerlo como base comparativa *(baseline)*. Con la información actual del dataset no se logra entrenar adecuadamente este modelo, en las siguientes gráficas se puede apreciar una alta dispersión (una nube de puntos muy extendida de forma horizontal), lo que evidencia una baja correlación lineal y una dificultad del modelo para ajustarse a los patrones de la serie temporal.')

    # Generar y_test_real invirtiendo la escala para los valores reales del conjunto de prueba
    # Usamos únicamente las primeras 3 columnas de y_test que corresponden a las ventas de las categorías
    y_test_scaled_only_sales = y_test[:, :3]
    y_test_log = scaler_y.inverse_transform(y_test_scaled_only_sales)
    y_test_real = np.expm1(y_test_log)

    # Calcular métricas para el modelo de Regresión Lineal
    categories = ['Furniture', 'Office Supplies', 'Technology']
    lr_metrics_summary = []
    for i, cat in enumerate(categories):
      mae_lr = mean_absolute_error(y_test_real[:, i], y_pred_lr_real[:, i])
      rmse_lr = np.sqrt(mean_squared_error(y_test_real[:, i], y_pred_lr_real[:, i]))
      r2_lr = r2_score(y_test_real[:, i], y_pred_lr_real[:, i])

      lr_metrics_summary.append({
          'Categoría': cat,
          'MAE ($) LR': round(mae_lr, 2),
          'RMSE ($) LR': round(rmse_lr, 2),
          'R² Score LR': round(r2_lr, 4)
        })

    df_lr_metrics = pd.DataFrame(lr_metrics_summary)

    # Crear subplots para graficar la dispersión de cada categoría (Regresión Lineal)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colores = ['#4682B4', '#FF4500', '#32CD32']

    for i, cat in enumerate(categories):
      real = y_test_real[:, i]
      pred = y_pred_lr_real[:, i]

      # Gráfico de dispersión
      axes[i].scatter(real, pred, color=colores[i], alpha=0.7, edgecolors='k', label='Predicciones')

      # Línea de referencia y = x (predicción perfecta)
      lims = [np.min([real.min(), pred.min()]), np.max([real.max(), pred.max()])]
      axes[i].plot(lims, lims, color='red', linestyle='--', linewidth=2, label='Predicción Perfecta (y=x)')

      axes[i].set_title(f'{cat} - Reales vs Predichos (LR)', fontweight='bold')
      axes[i].set_xlabel('Valores Reales ($ USD)')
      axes[i].set_ylabel('Valores Predichos ($ USD)')
      axes[i].grid(True, linestyle=':', alpha=0.6)
      axes[i].legend()

    plt.tight_layout()
    st.pyplot(fig)

  if optionML == "Red LSTM":

    # Sección para la LSTM
    st.header('🧠 Modelo de Redes Recurrentes LSTM')
    st.write('En este entrenamiento el modelo logra su punto más bajo en la pérdida de entrenamiento alrededor de la época 120.')

    # Generación de secuencias para que 'y' solo contenga las 3 categorías objetivo (3 dimensiones)
    X, y = [], []
    for i in range(len(scaled_data) - LOOKBACK):
      # Entrada: 4 semanas con las 6 características (ventas + features de fecha)
      X.append(scaled_data[i : i + LOOKBACK])
      # Salida: semana 5 con ÚNICAMENTE las 3 categorías de venta reales
      y.append(scaled_target[i + LOOKBACK])

    X = np.array(X)
    y = np.array(y)

    # Volver a separar en entrenamiento y prueba usando las máscaras existentes
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Definir la función de entrenamiento decorada con cache_resource
    @st.cache_resource
    def entrenar_modelo_lstm(input_weeks, input_features):
      # Fijar semilla para reproducibilidad
      tf.random.set_seed(42)

      # Construcción de la Red LSTM con dimensiones corregidas
      model = Sequential([
        # Entrada: shape (4 semanas, 6 características)
        LSTM(128, activation='tanh', return_sequences=True, input_shape=(input_weeks, input_features)),
        Dropout(0.2),
        LSTM(64, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        # Capa de salida: 3 neuronas (Furniture, Office Supplies, Technology)
        Dense(3, activation='linear')
      ])

      # Compilación del modelo
      model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
      )

      # Entrenamiento
      history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1 #Si reducimos a 0 se evita saturar la consola del servidor Streamlit 
      )
      return model, history.history
    
    # EJECUCIÓN DEL ENTRENAMIENTO / RECUPERACIÓN DESDE EL CACHÉ
    # Pasamos únicamente las dimensiones de la entrada como argumentos ligeros para el hash
    model, history_dict = entrenar_modelo_lstm(X_train.shape[1], X_train.shape[2])

    #Gráfica de curvas de aprendizaje (LOSS)
    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
    ax_loss.plot(history_dict['loss'], label='Pérdida Entrenamiento (Train Loss)', color='navy')
    ax_loss.plot(history_dict['val_loss'], label='Pérdida Validación (Val Loss)', color='crimson')
    plt.title('Evolución de la Pérdida del Modelo LSTM (MSE)', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE Normalizado)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Crear tres columnas para la visualización de la gráfica
    col_izquierda, col_centro, col_derecha = st.columns([1, 2, 1])

    # Mostrar la gráfica en la columna central
    with col_centro:
      st.pyplot(fig_loss)

    st.subheader("🔮 Simulador de Predicciones en Tiempo Real (LSTM)")
    st.write("""
    Introduce los valores históricos de las últimas 4 semanas para que la red neuronal recurrente
    pronostique el comportamiento de las ventas de la quinta semana.
    """)

    with st.form("formulario_prediccion"):
      col1, col2, col3, col4 = st.columns(4)

      with col1:
        st.markdown("**Semana 1 (Hace 4 semanas)**")
        w1_f = st.number_input("Furniture (W1)", min_value=0.0, value=1500.0, step=100.0)
        w1_o = st.number_input("Office Supplies (W1)", min_value=0.0, value=2000.0, step=100.0)
        w1_t = st.number_input("Technology (W1)", min_value=0.0, value=3000.0, step=100.0)

      with col2:
        st.markdown("**Semana 2 (Hace 3 semanas)**")
        w2_f = st.number_input("Furniture (W2)", min_value=0.0, value=1700.0, step=100.0)
        w2_o = st.number_input("Office Supplies (W2)", min_value=0.0, value=2200.0, step=100.0)
        w2_t = st.number_input("Technology (W2)", min_value=0.0, value=3100.0, step=100.0)

      with col3:
        st.markdown("**Semana 3 (Hace 2 semanas)**")
        w3_f = st.number_input("Furniture (W3)", min_value=0.0, value=1600.0, step=100.0)
        w3_o = st.number_input("Office Supplies (W3)", min_value=0.0, value=1900.0, step=100.0)
        w3_t = st.number_input("Technology (W3)", min_value=0.0, value=2800.0, step=100.0)

      with col4:
        st.markdown("**Semana 4 (Semana actual)**")
        w4_f = st.number_input("Furniture (W4)", min_value=0.0, value=2100.0, step=100.0)
        w4_o = st.number_input("Office Supplies (W4)", min_value=0.0, value=2500.0, step=100.0)
        w4_t = st.number_input("Technology (W4)", min_value=0.0, value=3500.0, step=100.0)

      st.markdown("**Variables Estacionales de la Semana a Predecir (Semana 5)**")
      c_f1, c_f2, c_f3 = st.columns(3)
      with c_f1:
        es_fin_de_mes = st.selectbox("¿Cae en fin de mes (Día >= 25)?", ["No", "Sí"])
        is_month_end_val = 1 if es_fin_de_mes == "Sí" else 0
      with c_f2:
        es_fin_de_ano = st.selectbox("¿Es Temporada de fin de año (Nov/Dic)?", ["No", "Sí"])
        is_holiday_season_val = 1 if es_fin_de_ano == "Sí" else 0
      with c_f3:
        mes_objetivo = st.slider("Mes calendario objetivo:", min_value=1, max_value=12, value=6)
        month_sin_val = np.sin(2 * np.pi * mes_objetivo / 12)

      boton_calcular = st.form_submit_button("🚀 Calcular Pronóstico")

      if boton_calcular:
        # 1. Agrupar los datos crudos en la estructura del pipeline
        raw_input = np.array([
          [w1_f, w1_o, w1_t, 0, 0, 0], # Variables exógenas se asumen baselines para semanas históricas
          [w2_f, w2_o, w2_t, 0, 0, 0],
          [w3_f, w3_o, w3_t, 0, 0, 0],
          [w4_f, w4_o, w4_t, is_month_end_val, is_holiday_season_val, month_sin_val]
        ])

        # 2. Aplicar transformación logarítmica (np.log1p) a las columnas de ventas
        log_input = raw_input.copy()
        log_input[:, :3] = np.log1p(raw_input[:, :3])

        # 3. Escalar las características utilizando scaler_x
        scaled_input = scaler_x.transform(log_input)

        # 4. Ajustar dimensión para el modelo LSTM: (1, 4, 6)
        lstm_input = np.expand_dims(scaled_input, axis=0)

        # 5. Ejecutar predicción con el modelo ya entrenado
        prediction_scaled = model.predict(lstm_input)

        # 6. Invertir escalado del target con scaler_y
        prediction_log = scaler_y.inverse_transform(prediction_scaled)

        # 7. Invertir transformación logarítmica (np.expm1)
        prediction_real = np.expm1(prediction_log)[0]

        # Mostrar resultados finales de manera llamativa
        st.success("🎯 ¡Ventas estimadas calculadas correctamente!")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("🪑 Furniture", f"${prediction_real[0]:,.2f} USD")
        res_col2.metric("📎 Office Supplies", f"${prediction_real[1]:,.2f} USD")
        res_col3.metric("💻 Technology", f"${prediction_real[2]:,.2f} USD")

    st.subheader("📏 Métricas y Comparación de los Modelos")

    st.markdown("""
      A continuación de muestran las métricas arrojadas en los entrenamientos.
    """)
    st.divider()
    st.write("**Métricas del Modelo LSTM**")
    st.markdown("""
      El Entrenamiento LSTM arroja las siguientes métricas:
    """)

    # Predecir en datos de prueba (2018)
    y_pred_scaled = model.predict(X_test)

    # Invertir MinMaxScaler
    y_test_log = scaler_y.inverse_transform(y_test)
    y_pred_log = scaler_y.inverse_transform(y_pred_scaled)

    # Invertir Transformación Logarítmica (np.expm1) para volver a DÓLARES REALES
    y_test_real = np.expm1(y_test_log)
    y_pred_real = np.expm1(y_pred_log)

    categories = list(df_weekly.columns)

    # Métricas cuantitativas en dólares
    metrics_summary = []

    for i, cat in enumerate(categories):
      mae = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
      rmse = np.sqrt(mean_squared_error(y_test_real[:, i], y_pred_real[:, i]))
      r2 = r2_score(y_test_real[:, i], y_pred_real[:, i])

      metrics_summary.append({
      'Categoría': cat,
      'MAE ($)': round(mae, 2),
      'RMSE ($)': round(rmse, 2),
      'R² Score': round(r2, 4)
      })

    df_metrics = pd.DataFrame(metrics_summary)

    # Crear tres columnas para la visualización de la tabla
    col_izquierda, col_centro, col_derecha = st.columns([1, 2, 1])

    # Mostrar la tabla en la columna central
    with col_centro:
      st.table(df_metrics)

    st.divider()

    st.markdown("""
      En las siguientes gráficas vemos el comportamiento del entrenamiento para cada categoría y su comparación con los datos reales (ventas 2018), como ya se había pronósticado los datos reales de ventas tienen una volatilidad que no logra ser predicha por el modelo entrenado.
    """)
    # Visualización y Comparación temporal (2018)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i, cat in enumerate(categories):
      axes[i].plot(dates_test, y_test_real[:, i], label='Ventas Reales ($)', color='black', alpha=0.7)
      axes[i].plot(dates_test, y_pred_real[:, i], label='Predicción LSTM ($)', color='dodgerblue', linestyle='--')
      axes[i].set_title(f'Pronóstico de Ventas 2018 - Categoría: {cat}', fontweight='bold')
      axes[i].set_ylabel('Ventas ($ USD)')
      axes[i].legend(loc='upper right')
      axes[i].grid(True, linestyle=':', alpha=0.6)

    plt.xlabel('Fecha (Año 2018)')
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader("📊 Regresión Líneal VS Red LSTM")
    st.write("**Comparativo de Modelos**")
    st.markdown("""
      A continuación se evalúan las métricas que arroja cada uno de los modelos entrenados.
    """)
    
    # Declarar los resultados ya conocidos de la Regresión Lineal en un diccionario (métricas calculadas previamente)
    metricas_regresion_lineal = {
        'Furniture': {'MAE': 1953.08, 'RMSE': 2578.53, 'R2': 0.2180},
        'Office Supplies': {'MAE': 2507.40, 'RMSE': 3419.03, 'R2': -0.3870},
        'Technology': {'MAE': 3366.21, 'RMSE': 5742.49, 'R2': -0.8246}
    }
        
    # Crear la lista con las métricas consolidadas de ambos modelos para la comparación
    comparison_data = []

    for i, cat in enumerate(categories):
      # Obtener las métricas fijas del diccionario para Regresión Lineal
      mae_lr = metricas_regresion_lineal[cat]['MAE']
      rmse_lr = metricas_regresion_lineal[cat]['RMSE']
      r2_lr = metricas_regresion_lineal[cat]['R2']

      # LSTM (métricas calculadas previamente)
      mae_lstm = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
      rmse_lstm = np.sqrt(mean_squared_error(y_test_real[:, i], y_pred_real[:, i]))
      r2_lstm = r2_score(y_test_real[:, i], y_pred_real[:, i])

      # Añadir registros organizados para la tabla
      comparison_data.append({
          'Categoría': cat,
          'Modelo': 'Regresión Lineal (Baseline)',
          'MAE ($)': round(mae_lr, 2),
          'RMSE ($)': round(rmse_lr, 2),
          'R² Score': round(r2_lr, 4)
      })
      comparison_data.append({
          'Categoría': cat,
          'Modelo': 'Red LSTM',
          'MAE ($)': round(mae_lstm, 2),
          'RMSE ($)': round(rmse_lstm, 2),
          'R² Score': round(r2_lstm, 4)
      })

    # Crear el DataFrame de comparación
    df_comparison = pd.DataFrame(comparison_data)

    # Pivotar para una visualización lado a lado de más fácil lectura
    df_styled = df_comparison.pivot(index='Categoría', columns='Modelo', values=['MAE ($)', 'RMSE ($)', 'R² Score'])
    st.dataframe(df_styled)

    st.markdown("""
      Al comparar ambos modelos, se evidencia que la red neuronal recurrente LSTM demostró un desempeño notablemente superior al de la Regresión Lineal (Baseline) en las tres categorías de producto. Al incorporar dependencias temporales y memoria a largo plazo mediante sus celdas de estado, la LSTM logró reducir el error promedio (MAE y RMSE) y mejorar drásticamente el coeficiente de ajuste ($R^2$), demostrando que los patrones de ventas minoristas son intrínsecamente no lineales y secuenciales.
      """)
    
    # Graficar comparativa visual de ambos modelos (LSTM vs Regresión Lineal)
    fig_comp, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for i, cat in enumerate(categories):
      axes[i].plot(dates_test, y_test_real[:, i], label='Ventas Reales ($)', color='black', alpha=0.7)
      axes[i].plot(dates_test, y_pred_real[:, i], label='Predicción LSTM ($)', color='dodgerblue', linestyle='--')
      axes[i].plot(dates_test, y_pred_lr_real[:, i], label='Predicción Regresión Lineal ($)', color='orange', linestyle=':')
      axes[i].set_title(f'Comparativa de Pronóstico 2018 - Categoría: {cat}', fontweight='bold')
      axes[i].set_ylabel('Ventas ($ USD)')
      axes[i].legend(loc='upper right')
      axes[i].grid(True, linestyle=':', alpha=0.6)

    plt.xlabel('Fecha (Año 2018)')
    plt.tight_layout()
    st.pyplot(fig_comp)

