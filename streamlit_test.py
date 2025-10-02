import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pickle

# Configuración de la página
st.set_page_config(page_title="AutoML con Métricas Avanzadas", layout="wide")

st.title("🤖 AutoML con Análisis AUC-ROC en Streamlit Cloud")
st.info("✅ Incluye gráficos de Área Bajo la Curva y métricas avanzadas")

# Función para calcular métricas y gráficos ROC
def plot_roc_curve(y_true, y_pred_proba, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # Curva ROC
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.3f})',
        line=dict(width=3, color='blue')
    ))
    
    # Línea de referencia (clasificador aleatorio)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Clasificador Aleatorio',
        line=dict(dash='dash', color='red', width=2)
    ))
    
    fig.update_layout(
        title=f'Curva ROC - {model_name}',
        xaxis_title='Tasa de Falsos Positivos (FPR)',
        yaxis_title='Tasa de Verdaderos Positivos (TPR)',
        width=600,
        height=500,
        showlegend=True
    )
    
    return fig, roc_auc

# Función para plot Precision-Recall
def plot_precision_recall_curve(y_true, y_pred_proba, model_name):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'{model_name} (AUC = {pr_auc:.3f})',
        line=dict(width=3, color='green')
    ))
    
    fig.update_layout(
        title=f'Curva Precision-Recall - {model_name}',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=600,
        height=500,
        showlegend=True
    )
    
    return fig, pr_auc

# Función para matriz de confusión
def plot_confusion_matrix_heatmap(y_true, y_pred, model_name):
    from sklearn.metrics import confusion_matrix
    import plotly.figure_factory as ff
    
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true))
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title=f'Matriz de Confusión - {model_name}',
        xaxis_title='Predicción',
        yaxis_title='Real',
        width=500,
        height=500
    )
    
    return fig

# Función para entrenar modelos con probabilidades
def train_automl_models_with_proba(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        with st.spinner(f"Entrenando {name}..."):
            try:
                # Escalar datos para modelos que lo requieran
                if name == 'Logistic Regression':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                
                # Calcular métricas adicionales
                from sklearn.metrics import precision_score, recall_score, f1_score
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'predictions_proba': y_pred_proba,
                    'scaler': scaler if name == 'Logistic Regression' else None
                }
                
            except Exception as e:
                st.warning(f"Modelo {name} tuvo un problema: {e}")
    
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
            st.write(f"**Tipos de datos:**")
            st.dataframe(df.dtypes.astype(str))
            
            # Información de la variable objetivo
            target_col = st.selectbox(
                "Selecciona la columna objetivo (target):",
                options=df.columns,
                index=len(df.columns)-1
            )
            
            if target_col:
                st.write(f"**Distribución de {target_col}:**")
                target_counts = df[target_col].value_counts()
                st.dataframe(target_counts)
        
        # Selección de características
        st.subheader("🔧 Configuración del Modelo")
        
        exclude_cols = st.multiselect(
            "Columnas a excluir (opcional):",
            options=df.columns,
            default=[target_col]
        )
        
        # Preparar datos
        feature_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]
        
        if len(feature_cols) == 0:
            st.error("❌ Debes seleccionar al menos una columna como feature")
        else:
            st.info(f"**Features:** {feature_cols}")
            st.info(f"**Target:** {target_col}")
            
            X = df[feature_cols]
            y = df[target_col]
            
            # Manejar variables categóricas en features
            X = pd.get_dummies(X, drop_first=True)
            
            # Codificar variable objetivo si es categórica
            le = None
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                st.info("✅ Variable objetivo codificada numéricamente")
            
            # Dividir datos
            test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.4, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            st.write(f"**Conjunto de entrenamiento:** {X_train.shape[0]} muestras")
            st.write(f"**Conjunto de prueba:** {X_test.shape[0]} muestras")
            
            # Entrenar modelos
            if st.button("🚀 Ejecutar AutoML con Análisis ROC"):
                with st.spinner("Entrenando modelos y generando métricas..."):
                    results = train_automl_models_with_proba(X_train, X_test, y_train, y_test)
                
                if not results:
                    st.error("❌ No se pudo entrenar ningún modelo")
                else:
                    # Mostrar leaderboard
                    st.subheader("📊 Leaderboard de Modelos")
                    
                    leaderboard_data = []
                    for name, result in results.items():
                        leaderboard_data.append({
                            'Modelo': name,
                            'Precisión': f"{result['accuracy']:.4f}",
                            'F1-Score': f"{result['f1_score']:.4f}",
                            'Precision': f"{result['precision']:.4f}",
                            'Recall': f"{result['recall']:.4f}"
                        })
                    
                    leaderboard_df = pd.DataFrame(leaderboard_data)
                    leaderboard_df = leaderboard_df.sort_values('Precisión', ascending=False)
                    
                    st.dataframe(leaderboard_df, use_container_width=True)
                    
                    # Mejor modelo
                    best_model_name = leaderboard_df.iloc[0]['Modelo']
                    best_result = results[best_model_name]
                    
                    st.success(f"🎉 **Mejor modelo:** {best_model_name}")
                    
                    # Mostrar métricas avanzadas en columnas
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Precisión", f"{best_result['accuracy']:.4f}")
                    with col2:
                        st.metric("F1-Score", f"{best_result['f1_score']:.4f}")
                    with col3:
                        st.metric("Precision", f"{best_result['precision']:.4f}")
                    with col4:
                        st.metric("Recall", f"{best_result['recall']:.4f}")
                    
                    # GRÁFICOS AVANZADOS
                    st.subheader("📈 Análisis Visual del Mejor Modelo")
                    
                    # Crear pestañas para diferentes gráficos
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Curva ROC", 
                        "🎯 Precision-Recall", 
                        "🔍 Matriz de Confusión",
                        "📋 Reporte Detallado"
                    ])
                    
                    with tab1:
                        st.header("Área Bajo la Curva ROC (AUC-ROC)")
                        st.info("""
                        **Interpretación de la Curva ROC:**
                        - **Área = 1.0**: Clasificador perfecto
                        - **Área > 0.9**: Excelente
                        - **Área > 0.8**: Muy bueno  
                        - **Área > 0.7**: Aceptable
                        - **Área = 0.5**: Clasificador aleatorio
                        """)
                        
                        # Gráfico ROC para el mejor modelo
                        fig_roc, roc_auc = plot_roc_curve(
                            y_test, best_result['predictions_proba'], best_model_name
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                        
                        # Mostrar AUC value
                        st.metric("Área Bajo la Curva ROC (AUC)", f"{roc_auc:.4f}")
                        
                        # Comparar con otros modelos
                        st.subheader("Comparación con Otros Modelos")
                        roc_fig_comparison = go.Figure()
                        roc_fig_comparison.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Clasificador Aleatorio',
                            line=dict(dash='dash', color='red', width=2)
                        ))
                        
                        for name, result in results.items():
                            fpr, tpr, _ = roc_curve(y_test, result['predictions_proba'])
                            model_auc = auc(fpr, tpr)
                            roc_fig_comparison.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=f'{name} (AUC = {model_auc:.3f})',
                                line=dict(width=2)
                            ))
                        
                        roc_fig_comparison.update_layout(
                            title='Comparación de Curvas ROC - Todos los Modelos',
                            xaxis_title='Tasa de Falsos Positivos (FPR)',
                            yaxis_title='Tasa de Verdaderos Positivos (TPR)',
                            width=800,
                            height=600
                        )
                        st.plotly_chart(roc_fig_comparison, use_container_width=True)
                    
                    with tab2:
                        st.header("Curva Precision-Recall")
                        fig_pr, pr_auc = plot_precision_recall_curve(
                            y_test, best_result['predictions_proba'], best_model_name
                        )
                        st.plotly_chart(fig_pr, use_container_width=True)
                        st.metric("Área Bajo Curva Precision-Recall", f"{pr_auc:.4f}")
                    
                    with tab3:
                        st.header("Matriz de Confusión")
                        fig_cm = plot_confusion_matrix_heatmap(
                            y_test, best_result['predictions'], best_model_name
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with tab4:
                        st.header("Reporte de Clasificación Detallado")
                        st.text(classification_report(y_test, best_result['predictions']))
                        
                        # Características importantes (si el modelo lo soporta)
                        if hasattr(best_result['model'], 'feature_importances_'):
                            st.subheader("🔝 Importancia de Características")
                            feature_importance = pd.DataFrame({
                                'feature': X.columns,
                                'importance': best_result['model'].feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            fig_importance = px.bar(
                                feature_importance.head(10),
                                x='importance',
                                y='feature',
                                orientation='h',
                                title='Top 10 Características Más Importantes'
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Descargar modelo
                    st.subheader("💾 Guardar Modelo")
                    model_data = {
                        'model': best_result['model'],
                        'feature_names': feature_cols,
                        'target_name': target_col,
                        'label_encoder': le,
                        'scaler': best_result.get('scaler', None),
                        'metrics': {
                            'accuracy': best_result['accuracy'],
                            'roc_auc': roc_auc,
                            'pr_auc': pr_auc
                        }
                    }
                    
                    model_bytes = pickle.dumps(model_data)
                    st.download_button(
                        label="📥 Descargar Mejor Modelo",
                        data=model_bytes,
                        file_name=f"mejor_modelo_{best_model_name}.pkl",
                        mime="application/octet-stream"
                    )
                    
    except Exception as e:
        st.error(f"❌ Error procesando los datos: {e}")
        st.error(f"Detalles: {str(e)}")

# Ejemplo con datos de prueba
else:
    st.header("💡 Ejemplo con Datos de Prueba")
    
    if st.button("Cargar Dataset de Ejemplo - Breast Cancer"):
        from sklearn.datasets import load_breast_cancer
        
        data = load_breast_cancer()
        df_example = pd.DataFrame(data.data, columns=data.feature_names)
        df_example['target'] = data.target
        
        st.session_state.example_data = df_example
        st.rerun()
    
    if 'example_data' in st.session_state:
        st.dataframe(st.session_state.example_data.head())
        st.info("""
        **Dataset de Cáncer de Mama:**
        - 30 características numéricas
        - Variable objetivo binaria (0: maligno, 1: benigno)
        - Perfecto para probar curvas ROC
        """)

# Información adicional
with st.expander("ℹ️ Guía de Interpretación de Métricas"):
    st.markdown("""
    **📊 Curva ROC (Receiver Operating Characteristic):**
    - **AUC = 1.0**: Clasificador perfecto
    - **AUC > 0.9**: Excelente discriminación
    - **AUC > 0.8**: Muy buena discriminación
    - **AUC > 0.7**: Discriminación aceptable
    - **AUC = 0.5**: No mejor que adivinar al azar
    
    **🎯 Curva Precision-Recall:**
    - Más informativa que ROC cuando las clases están desbalanceadas
    - Muestra el trade-off entre precision y recall
    
    **🔍 Matriz de Confusión:**
    - **Verdaderos Positivos (TP)**: Correctamente clasificados como positivos
    - **Falsos Positivos (FP)**: Incorrectamente clasificados como positivos
    - **Verdaderos Negativos (TN)**: Correctamente clasificados como negativos
    - **Falsos Negativos (FN)**: Incorrectamente clasificados como negativos
    """)