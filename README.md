
# ⚽ AI Football Predictor: Predice Resultados con Inteligencia Artificial

---

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI Football Predictor** es una herramienta impulsada por Inteligencia Artificial capaz de predecir el resultado de partidos de fútbol en base a estadísticas reales. Está diseñada como un proyecto educativo, ideal para quienes desean aprender Machine Learning aplicado al deporte y trabajar con datos reales de fútbol.

---

## ✨ Características

- 🔍 **Procesamiento de Datos**: Limpieza y transformación de datos estadísticos de partidos.
- 🧠 **Entrenamiento de Modelos de IA**: Utiliza algoritmos de clasificación para predecir el resultado (victoria, empate, derrota).
- 📊 **Evaluación y Métricas**: Métricas clave como precisión, matriz de confusión, etc.
- 💾 **Predicción con Nuevos Datos**: Puedes ingresar nuevas estadísticas de un partido y obtener la predicción.
- 📁 **Estructura Modular**: Proyecto dividido por responsabilidades para facilitar mantenimiento y comprensión.

---

## 📂 Estructura del Proyecto

```
futbol_ia/
├── datos/
│   └── partidos.csv               # Dataset principal con estadísticas de partidos históricos
├── modelos/
│   └── modelo_entrenado.pkl       # Modelo entrenado y serializado con joblib
├── scripts/
│   ├── cargar_datos.py            # Funciones para cargar y preparar los datos
│   ├── entrenar_modelo.py         # Entrena el modelo de clasificación
│   ├── predecir.py                # Utiliza el modelo para predecir nuevos partidos
├── README.md                      # Este archivo
├── requirements.txt               # Dependencias necesarias
└── main.py                        # Script principal de ejecución
```

---

## ⚙️ Requisitos Previos

- Python 3.10+
- pip (gestor de paquetes)
- Bibliotecas del archivo `requirements.txt`:
    ```txt
    pandas
    numpy
    scikit-learn
    joblib
    ```

---

## 🚀 Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/tu_usuario/futbol_ia.git
cd futbol_ia
```

2. Crea y activa un entorno virtual:

```bash
python -m venv venv
# En Windows
venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución Paso a Paso

### 1. Entrenar el modelo

```bash
python scripts/entrenar_modelo.py
```

Esto cargará los datos desde `datos/partidos.csv`, entrenará el modelo y lo guardará en `modelos/modelo_entrenado.pkl`.

### 2. Predecir un nuevo partido

Edita el archivo `scripts/predecir.py` con los datos del partido a predecir, luego ejecútalo:

```bash
python scripts/predecir.py
```

Ejemplo de entrada en `predecir.py`:
```python
nuevo_partido = {
    'posesion': 55,
    'tiros': 10,
    'tiros_a_puerta': 5,
    'corners': 4,
    'faltas': 12,
    ...
}
```

---

## 📈 Evaluación del Modelo

Durante el entrenamiento, el sistema imprime:

- Precisión (`accuracy`)
- Matriz de confusión
- Informe de clasificación (`classification_report`)

Estos indicadores permiten saber cómo de bien está funcionando el modelo con los datos de prueba.

---

## 🧠 Algoritmos Utilizados

- `RandomForestClassifier` (clasificador principal)
- Posibilidad de extender a otros modelos como:
    - `LogisticRegression`
    - `GradientBoostingClassifier`
    - `XGBoost` (con instalación adicional)

---

## 💡 Personalización

Puedes mejorar y personalizar el proyecto:

- Usar más características (e.g., tarjetas, localía, alineación).
- Cambiar el algoritmo y comparar resultados.
- Añadir una interfaz web con Flask o Streamlit.
- Conectar a una API de fútbol como [Football-Data.org](https://www.football-data.org/) para obtener datos en tiempo real.

---

## 🧪 Ejemplo de Predicción

```bash
Resultado predicho: ⚽ ¡Victoria del equipo local!
```

---

## ❗ Posibles Errores Comunes

- **FileNotFoundError**: Verifica que el archivo `partidos.csv` esté en la carpeta `datos/`.
- **ImportError**: Asegúrate de instalar todos los paquetes del `requirements.txt`.
- **Modelo no entrenado**: Ejecuta primero `entrenar_modelo.py` antes de predecir.

---

## 👨‍💻 Autor

Proyecto desarrollado por Stefan Mario Magura, apasionado por la IA y el fútbol.

📧 Email: stefanmariomagura@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/stefan-mario-magura-290a26367/

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
