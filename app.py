import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Income Prediction ML",
    page_icon="📊",
    layout="wide")

# -------------------------------------------------------------
# Sidebar menu
# -------------------------------------------------------------
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Selecciona una sección:",
    ["Introducción",
        "Dataset",
        "EDA",
        "Preprocesado",
        "Modelos probados",
        "Modelo Final",
        "Importancia de Variables",
        "Demo: Predicción en vivo",
        "Sesgos y Ética",
        "Conclusiones",
        "Enfoque"])

# -------------------------------------------------------------
# Helper: load dataset
# -------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data/adult.data.xlsx")
        return df
    except Exception as e:
        return None

df = load_data()

# -------------------------------------------------------------
# INTRODUCCIÓN
# -------------------------------------------------------------
if menu == "Introducción":
    st.title("¿Gana una persona más de 50k$/año?")
    st.subheader("Proyecto de Machine Learning")
    st.write("""
    Esta aplicación muestra de forma interactiva si una persona gana más de 50k$/año utilizando datos censales reales de EE.UU.
    """)

    st.markdown("""
    ### Objetivos:  
    - Predecir si una persona gana más de 50.000$/año  
    - Aplicar técnicas de limpieza y preprocesado de datos  
    - Entrenar y comparar varios modelos de clasificación  
    - Tratar el desbalanceo de clases con SMOTE  
    - Seleccionar un modelo final y evaluarlo  
    - Reflexionar sobre los sesgos y la ética en ML  
    """)

# -------------------------------------------------------------
# DATASET
# -------------------------------------------------------------
elif menu == "Dataset":
    st.title("Dataset – Adult Census Income")
    if df is None:
        st.error("No se pudo cargar el dataset. Asegúrate de que 'adult.data.xlsx' está en la misma carpeta que app.py.")
    else:
        st.write("Vista previa del dataset:")
        st.dataframe(df.head())

        st.write("Dimensiones del dataset:")
        st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

# -------------------------------------------------------------
# EDA
# -------------------------------------------------------------
elif menu == "EDA":
    st.title("Análisis Exploratorio de Datos")

    if df is None:
        st.error("No se pudo cargar el dataset.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribución de la Edad")
            fig1 = plt.figure(figsize=(6,4))
            sns.histplot(df["age"], bins=30, kde=True)
            plt.xlabel("Edad")
            plt.ylabel("Frecuencia")
            st.pyplot(fig1)

        with col2:
            st.subheader("Horas trabajadas por semana")
            fig2 = plt.figure(figsize=(6,4))
            sns.histplot(df["hours-per-week"], bins=30, kde=True)
            plt.xlabel("Horas/semana")
            plt.ylabel("Frecuencia")
            st.pyplot(fig2)

        st.subheader("Nivel educativo según income")
        fig3 = plt.figure(figsize=(8,6))
        sns.countplot(y="education", hue="income", data=df,
                      order=df["education"].value_counts().index)
        plt.xlabel("Número de personas")
        plt.ylabel("Nivel educativo")
        st.pyplot(fig3)

# -------------------------------------------------------------
# PREPROCESADO
# -------------------------------------------------------------
elif menu == "Preprocesado":
    st.title("Preparación y Limpieza de Datos")

    st.markdown("""
    Pasos realizados:

    1. Eliminación de duplicados  
    2. Sustitución de valores "?" por NaN  
    3. Eliminación de filas con valores faltantes  
    4. Codificación One-Hot de variables categóricas  
    5. Estandarización de variables numéricas con StandardScaler  
    6. División del dataset en train/test (80/20) con estratificación  
    7. Rebalanceo del conjunto de entrenamiento con **SMOTE** (Solo el 24% de los registros corresponden a ingresos >50K) 
    """)

# -------------------------------------------------------------
# MODELOS PROBADOS
# -------------------------------------------------------------
elif menu == "Modelos probados":
    st.title("Modelos probados durante el proyecto")

    st.markdown("""
    Se entrenaron y evaluaron los siguientes modelos de clasificación:

    - Regresión Logística  
    - K-Nearest Neighbors (KNN)  
    - Árbol de Decisión  
    - Random Forest  
    - Gradient Boosting  
    - AdaBoost  

    La comparación se hizo usando métricas como:
    - Accuracy  
    - Precision  
    - Recall  
    - F1-score  
    """)

# -------------------------------------------------------------
# MODELO FINAL
# -------------------------------------------------------------
elif menu == "Modelo Final":
    st.title("Modelo Final: Gradient Boosting")

    st.markdown("""
    Tras comparar los modelos y aplicar **RandomizedSearchCV** para ajustar hiperparámetros, 
    el modelo seleccionado fue:

    ### Gradient Boosting Classifier

    Características:
    - Buen equilibrio entre precisión y recall
    - Mejor F1-score para la clase `>50K`
    - Estable tras el rebalanceo con SMOTE
    """)

    st.subheader("Matriz de Confusión")
    try:
        st.image("images/matriz_confusion.png", caption="Matriz de Confusión del modelo final")
    except:
        st.info("Añade 'matriz_confusion.png' en la misma carpeta que app.py.")

# -------------------------------------------------------------
# IMPORTANCIA DE VARIABLES
# -------------------------------------------------------------
elif menu == "Importancia de Variables":
    st.title("Variables más importantes")

    st.markdown("""
    Según el modelo final de Gradient Boosting, las variables más importantes fueron, entre otras:

    - capital-gain  
    - education-num  
    - age  
    - hours-per-week  
    - marital-status  
    """)

    try:
        st.image("images/importancia_variables.png", caption="Top 10 variables más importantes")
    except:
        st.info("Añade 'importancia_variables.png' en la misma carpeta que app.py.")

# -------------------------------------------------------------
# ÉTICA
# -------------------------------------------------------------
elif menu == "Sesgos y Ética":
    st.title("Consideraciones Éticas y Posibles Sesgos")

    st.markdown("""
    El dataset incluye variables sensibles como **sexo** y **raza**, lo que puede introducir sesgos en el modelo.

    - Los hombres aparecen con más probabilidad de ingresos >50K  
    - Algunos grupos raciales están sobrerrepresentados o infrarrepresentados  

    En una aplicación real, esto podría amplificar desigualdades existentes.

    Recomendaciones:
    - Analizar métricas por subgrupos 
    - Considerar excluir o anonimizar variables sensibles  
    - Auditar los modelos con regularidad  
    """)

# -------------------------------------------------------------
# CONCLUSIONES
# -------------------------------------------------------------
elif menu == "Conclusiones":
    st.title("Conclusiones")

    st.markdown("""
    - Es posible predecir si una persona gana >50K$/año con un rendimiento sólido.  
    - El preprocesado y el tratamiento del desbalanceo son claves para obtener buenos resultados.  
    - El modelo de **Gradient Boosting** ofrece el mejor equilibrio entre métricas.  
    - Las variables de educación, capital-gain, edad y horas trabajadas resultan especialmente relevantes.  
    - Es fundamental tener en cuenta los sesgos de los datos antes de aplicar el modelo en entornos reales.  
    """)

# -------------------------------------------------------------
# CONCLUSIONES
# -------------------------------------------------------------
elif menu == "Enfoque":
    st.title("Enfoque empresarial")

    st.markdown("""
    Este modelo puede ayudar a empresas, consultoras o instituciones a entender mejor los factores que influencian los ingresos de una persona. 
    
    Esto permite:

    - Mejor segmentación de clientes
    - Diseño de campañas de marketing más efectivas
    - Identificación de perfiles con mayor poder adquisitivo
    - Optimización de estrategias de captación y retención
    - Análisis de riesgo socioeconómico
     """)


# -------------------------------------------------------------
# DEMO INTERACTIVA
# -------------------------------------------------------------
elif menu == "Demo: Predicción en vivo":
    st.title("Demo: Predicción en vivo")
    st.write("Introduce los datos reales de la persona para estimar si ganará más o menos de 50K$/año.")

    try:
        model = joblib.load("models/best_model.pkl")
        columnas_modelo = joblib.load("models/columnas_modelo.pkl")

        # =======================
        #   SELECCIÓN DE INPUTS
        # =======================
        st.subheader("Características de la persona")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Edad", 17, 90, 30)
            hours = st.slider("Horas trabajadas por semana", 1, 99, 40)
            sex = st.selectbox("Sexo", sorted(df["sex"].unique()))
            race = st.selectbox("Origen étnico", sorted(df["race"].unique()))

        with col2:
            education = st.selectbox("Nivel educativo", sorted(df["education"].unique()))
            marital_status = st.selectbox("Estado civil", sorted(df["marital-status"].unique()))
            occupation = st.selectbox("Ocupación", sorted(df["occupation"].unique()))
            native_country = st.selectbox("País de origen", sorted(df["native-country"].unique()))

        # ============================
        #     BOTÓN DE PREDICCIÓN
        # ============================
        if st.button("Predecir ingresos"):
            X_input = pd.DataFrame(np.zeros((1, len(columnas_modelo))), columns=columnas_modelo)

            if "age" in X_input.columns: X_input["age"] = age
            if "hours-per-week" in X_input.columns: X_input["hours-per-week"] = hours

            cat_values = {
                "education": education,
                "marital-status": marital_status,
                "occupation": occupation,
                "race": race,
                "sex": sex,
                "native-country": native_country
            }

            for col_prefix, value in cat_values.items():
                col_name = f"{col_prefix}_{value}"
                if col_name in X_input.columns:
                    X_input[col_name] = 1

            # ============================
            #      PREDICCIÓN FINAL
            # ============================
            pred = model.predict(X_input)[0]

            st.subheader("Resultado de la predicción:")
            if pred == 1:
                st.success("Esta persona probablemente gana **más de 50K dólares anuales**.")
            else:
                st.error("Esta persona probablemente gana **50K o menos**.")

    except Exception as e:
        st.error("Error cargando el modelo o las columnas. Asegúrate de que:")
        st.write("- best_model.pkl está en la carpeta")
        st.write("- columnas_modelo.pkl está en la carpeta")
        st.write("- El modelo fue exportado correctamente")
        st.write(e)

