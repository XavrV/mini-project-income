# 🧑‍💻 Mini Project Income – Instrucciones de Uso

¡Bienvenido/a! Este proyecto es **totalmente reproducible y portátil**.

---

## 🚀 Pasos para ejecutar este proyecto

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/XavrV/mini-project-income.git
   ```
2. **(Muy importante)**
   Mueve este notebook dentro de la carpeta clonada (`mini-project-income/`)
   *o* ábrelo directamente desde ahí si ya está incluido en el repo.
3. **Instala las dependencias antes de ejecutar el notebook:**

   * **Conda (recomendado):**

     ```bash
     conda env create -f environment.yml
     conda activate mini-project-income
     ```
   * **O con pip (si usas un env):**

     ```bash
     pip install -r requirements.txt
     ```
4. **El dataset `adult.csv` se descarga automáticamente al ejecutar la celda inicial del notebook.**
   Si lo prefieres, puedes descargarlo manualmente con:

   ```bash
   wget https://raw.githubusercontent.com/pooja2512/Adult-Census-Income/master/adult.csv -O data/adult.csv --quiet
   ```
5. **¡Ya puedes correr el notebook!**
   Todos los módulos y funciones personalizadas están en `src/` y se importan automáticamente.


**¿Dudas o problemas?**
Revisa el README, los comentarios del notebook, o abre un Issue en el repo.

---

Este proyecto contiene un pipeline modular de análisis y modelado con *scikit-learn* para el dataset **Adult Census Income**. Cada bloque funcional del pipeline se ha separado en módulos reutilizables para facilitar el mantenimiento y la experimentación.

## Estructura general

- **`src/`** – módulos del pipeline (carga y validación, limpieza, ingeniería de variables, partición de datos, construcción del modelo y experimentación).
- **`data/`** – carpeta donde se colocará el dataset `adult.csv` y otros artefactos generados.
- **`main.py`** – script que orquesta todo el flujo de principio a fin.
- **Notebooks** – `01_pipeline_explorable.ipynb` y `mini_project_adult_income.ipynb` permiten explorar el pipeline de forma interactiva.


## Uso rápido

Una vez descargados los datos y con el entorno configurado, podemos lanzar el pipeline completo con:

```bash
python main.py
```

El script realiza la carga, limpieza, ingeniería de características, división de datos, construcción del modelo y una búsqueda de hiperparámetros registrada con MLflow.

## Próximos pasos

Explora los notebooks para ver ejemplos de EDA y cómo se componen las distintas etapas del pipeline.  Modifica `src/config.py` para ajustar columnas, algoritmos o parámetros y continúa experimentando.

