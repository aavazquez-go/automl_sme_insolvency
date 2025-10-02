import streamlit as st
import sys
import subprocess

# Intentar importar h2o, si no está instalado, mostrar instrucciones
try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    st.error("❌ H2O no está instalado")

import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(page_title="H2O en Streamlit", layout="wide")

st.title("🚀 Integración de H2O con Streamlit")

# Si H2O no está disponible, mostrar instrucciones de instalación
if not H2O_AVAILABLE:
    st.error("""
    **H2O no está instalado en este entorno.**
    
    Para instalar H2O, ejecuta:
    ```bash
    pip install h2o
    ```
    
    O crea un archivo `requirements.txt` con:
    ```txt
    streamlit>=1.28.0
    h2o>=3.42.0
    pandas>=1.5.0
    ```
    """)
    
    # Opción para intentar instalar automáticamente (solo funciona en algunos entornos)
    if st.button("Intentar instalar H2O automáticamente"):
        with st.spinner("Instalando H2O..."):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "h2o"])
                st.success("✅ H2O instalado correctamente. Recarga la página.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al instalar H2O: {e}")
    
    st.stop()

# Si H2O está disponible, continuar con la app normal
def init_h2o():
    try:
        h2o.init()
        return True
    except Exception as e:
        st.error(f"Error al inicializar H2O: {e}")
        return False

# Mostrar estado de H2O
st.header("🔧 Configuración de H2O")
if st.button("Inicializar H2O"):
    if init_h2o():
        st.success("✅ H2O inicializado correctamente")

# Resto de tu aplicación...
st.header("📊 Cargar y Procesar Datos")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Leer datos con pandas
        df = pd.read_csv(uploaded_file)
        st.write("Vista previa de los datos:")
        st.dataframe(df.head())
        
        # Convertir a frame de H2O
        h2o_df = h2o.H2OFrame(df)
        
        st.write("Frame de H2O:")
        st.write(h2o_df.head())
        
    except Exception as e:
        st.error(f"Error procesando los datos: {e}")

# Información del sistema
with st.expander("ℹ️ Información del sistema"):
    st.write(f"Python version: {sys.version}")
    st.write(f"H2O version: {h2o.__version__ if H2O_AVAILABLE else 'No disponible'}")