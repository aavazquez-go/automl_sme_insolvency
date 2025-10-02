import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

# Configuración de la página
st.set_page_config(page_title="AutoML Alternativo", layout="wide")

st.title("🤖 AutoML con scikit-learn en Streamlit Cloud")
st.info("✅ Esta solución funciona en Streamlit Cloud sin requerir Java")

# Función para entrenar múltiples modelos
def train_automl_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        with st.spinner(f"Entrenando {name}..."):
            # Escalar datos para Logistic Regression
            if name == 'Logistic Regression':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
    
    return results

# Interfaz principal
st.header("📊 Cargar Datos")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Leer datos
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Mostrar datos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vista previa")
            st.dataframe(df.head())
        
        with col2:
            st.subheader("Información de columnas")
            st.write(f"**Columnas:** {list(df.columns)}")
            st.write(f"**Tipo de datos:**")
            st.dataframe(df.dtypes.astype(str))
        
        # Selección de características y target
        st.subheader("🔧 Configuración del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox(
                "Selecciona la columna objetivo (target):",
                options=df.columns,
                index=len(df.columns)-1
            )
        
        with col2:
            exclude_cols = st.multiselect(
                "Columnas a excluir (opcional):",
                options=df.columns,
                default=[target_col]
            )
        
        # Preparar datos
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            st.error("❌ Debes seleccionar al menos una columna como feature")
        else:
            st.info(f"**Features:** {feature_cols}")
            st.info(f"**Target:** {target_col}")
            
            X = df[feature_cols]
            y = df[target_col]
            
            # Manejar variables categóricas
            X = pd.get_dummies(X, drop_first=True)
            
            # Dividir datos
            test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.4, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            st.write(f"**Conjunto de entrenamiento:** {X_train.shape[0]} muestras")
            st.write(f"**Conjunto de prueba:** {X_test.shape[0]} muestras")
            
            # Entrenar modelos
            if st.button("🚀 Ejecutar AutoML"):
                with st.spinner("Entrenando modelos..."):
                    results = train_automl_models(X_train, X_test, y_train, y_test)
                
                # Mostrar resultados
                st.subheader("📊 Resultados del AutoML")
                
                # Leaderboard
                leaderboard_data = []
                for name, result in results.items():
                    leaderboard_data.append({
                        'Modelo': name,
                        'Precisión': f"{result['accuracy']:.4f}"
                    })
                
                leaderboard_df = pd.DataFrame(leaderboard_data)
                leaderboard_df = leaderboard_df.sort_values('Precisión', ascending=False)
                
                st.dataframe(leaderboard_df, use_container_width=True)
                
                # Mejor modelo
                best_model_name = leaderboard_df.iloc[0]['Modelo']
                best_accuracy = leaderboard_df.iloc[0]['Precisión']
                
                st.success(f"🎉 **Mejor modelo:** {best_model_name} (Precisión: {best_accuracy})")
                
                # Reporte detallado del mejor modelo
                st.subheader(f"📈 Reporte detallado - {best_model_name}")
                best_result = results[best_model_name]
                
                st.text(classification_report(y_test, best_result['predictions']))
                
                # Descargar modelo
                st.subheader("💾 Guardar Modelo")
                model_bytes = pickle.dumps(best_result['model'])
                st.download_button(
                    label="Descargar mejor modelo",
                    data=model_bytes,
                    file_name=f"mejor_modelo_{best_model_name}.pkl",
                    mime="application/octet-stream"
                )
                
    except Exception as e:
        st.error(f"❌ Error procesando los datos: {e}")

# Ejemplo con datos de prueba
else:
    st.header("💡 Ejemplo con Datos de Prueba")
    
    if st.button("Cargar Dataset de Ejemplo - Iris"):
        from sklearn.datasets import load_iris
        
        iris = load_iris()
        df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
        df_iris['species'] = iris.target
        
        st.session_state.example_data = df_iris
        st.rerun()
    
    if 'example_data' in st.session_state:
        st.dataframe(st.session_state.example_data.head())
        st.info("✅ Datos de ejemplo cargados. Ahora puedes configurar y entrenar modelos.")

# Información adicional
with st.expander("ℹ️ Información sobre esta implementación"):
    st.markdown("""
    **¿Por qué esta solución funciona en Streamlit Cloud?**
    - ✅ No requiere Java
    - ✅ Solo usa bibliotecas Python puras
    - ✅ Compatible con las restricciones de Streamlit Cloud
    
    **Características incluidas:**
    - 📊 Carga y exploración de datos
    - 🔧 Selección automática de features
    - 🤖 Entrenamiento múltiple de modelos
    - 📈 Evaluación y comparación
    - 💾 Descarga de modelos entrenados
    
    **Modelos implementados:**
    - Random Forest
    - Gradient Boosting
    - Logistic Regression
    """)