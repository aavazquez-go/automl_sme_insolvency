import streamlit as st
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(page_title="H2O en Streamlit", layout="wide")

st.title("🚀 Integración de H2O con Streamlit")

# Inicializar H2O
@st.cache_resource
def init_h2o():
    h2o.init()
    return True

# Mostrar estado de H2O
if init_h2o():
    st.success("✅ H2O inicializado correctamente")

# Cargar datos de ejemplo
st.header("📊 Cargar Datos")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])

if uploaded_file is not None:
    # Leer datos con pandas
    df = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())
    
    # Convertir a frame de H2O
    h2o_df = h2o.H2OFrame(df)
    
    st.write("Frame de H2O:")
    st.write(h2o_df.head())
    
    # Ejemplo de modelo simple
    st.header("🤖 Entrenar Modelo con H2O")
    
    if st.button("Entrenar Modelo AutoML"):
        with st.spinner("Entrenando modelo..."):
            # Identificar predictores y variable objetivo
            predictors = h2o_df.columns[:-1]  # Todas menos la última
            response = h2o_df.columns[-1]     # Última columna
            
            # Crear y entrenar modelo AutoML
            aml = H2OAutoML(max_models=3, seed=1, max_runtime_secs=60)
            aml.train(x=predictors, y=response, training_frame=h2o_df)
            
            # Mostrar leaderboard
            st.subheader("Leaderboard del Modelo")
            lb = aml.leaderboard
            st.dataframe(lb.as_data_frame())
            
            st.success(f"🎉 Mejor modelo: {aml.leader.model_id}")

# Ejemplo con datos predefinidos
else:
    st.header("💡 Ejemplo con Datos de Prueba")
    
    if st.button("Usar Datos de Iris"):
        # Cargar dataset de iris
        from sklearn.datasets import load_iris
        iris = load_iris()
        df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
        df_iris['target'] = iris.target
        
        st.write("Dataset Iris cargado:")
        st.dataframe(df_iris.head())
        
        # Convertir a H2O frame
        h2o_iris = h2o.H2OFrame(df_iris)
        
        # Entrenar modelo
        with st.spinner("Entrenando modelo con Iris..."):
            predictors = h2o_iris.columns[:-1]
            response = h2o_iris.columns[-1]
            
            aml = H2OAutoML(max_models=2, seed=1, max_runtime_secs=30)
            aml.train(x=predictors, y=response, training_frame=h2o_iris)
            
            st.success(f"Modelo entrenado! Líder: {aml.leader.model_id}")

# Información del cluster H2O
st.header("🔍 Información del Cluster H2O")
if st.button("Mostrar información del cluster"):
    cluster_info = h2o.cluster_status()
    st.json(cluster_info)

# Cerrar H2O al final (opcional)
if st.button("Cerrar H2O"):
    h2o.cluster().shutdown()
    st.info("Cluster H2O cerrado")