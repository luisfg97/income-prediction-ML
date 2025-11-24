# 📘 Income Prediction ML

Proyecto del módulo de **Machine Learning (Ironhack)** para predecir si una persona gana **más de 50K$/año** usando el *Adult Income Dataset*.

---

## 🎯 Objetivo
Construir un modelo de clasificación capaz de estimar ingresos >50K a partir de variables como edad, educación, ocupación, horas trabajadas, sexo, origen étnico, etc.

---

## 📁 Dataset
- Fuente: Adult Census Income (UCI).  
- ~32.561 registros, 15 variables.  
- Procesado: limpieza, One-Hot Encoding, escalado, split 80/20 y **SMOTE** para balancear.

---

## 🤖 Modelos probados
- Regresión Logística, KNN, Decision Tree  
- Random Forest, Gradient Boosting, AdaBoost  

➡️ **Mejor modelo:** **Gradient Boosting**, optimizado con *RandomizedSearchCV*.

---

## 📊 Resultados
- Buen rendimiento en ambas clases.  
- Alta capacidad para identificar ingresos >50K.  
- Variables más influyentes: `capital-gain`, `education-num`, `age`, `hours-per-week`.  
- Gráficos disponibles en `/images`.

---

## ⚠️ Sesgos y ética
El dataset incluye variables sensibles (sexo, origen étnico).  
Requiere un uso responsable para evitar decisiones discriminatorias.

---

## 🧪 App interactiva (Streamlit)
Incluye demo de predicción y visualización del proyecto.

Ejecutar:

```
streamlit run app.py
```

---

## 🧱 Estructura

```
app.py
Income_pred_ml.ipynb
data/
models/
images/
README.md
```

---

## 👤 Autor

**Luis Fernández — Ironhack 2025**
