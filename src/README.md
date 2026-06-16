# Documentación de la Arquitectura de Software
## 1. Descripción General del código
El directorio /src contiene la implementación en Programación Orientada a Objetos (POO) para este Proyecto FInal de los modelos de riesgo crediticio. Su objetivo es que, a partir de los hallazgos matemáticos y análisis profundo de los Jupyter Notebooks, se migre hacia un entorno de producción automatizado (MLOps).
A diferencia de un script lineal, este código está diseñado bajo principios de ingeniería de software (tales como la separación de responsabilidades y bajo acoplamiento). Las funciones principales son:
* Automatización: Ejecutar toda la tubería de datos o *pipeline* (limpieza, partición, escalado, entrenamiento y evaluación) de maneta automatizada.
* Reproducibilidad: Garantizar que las transformaciones estadísticas se aprendan exclusivamente del conjunto de entrenamiento y se apliquen de forma consistente a datos nuevos.
* Persistencia: Generar y guardar automáticamente los artefactos del modelo (.joblib y .pth) para su posterior uso.

## 2. Estructura del directorio
La estructura del código está organizando los archivos en capas lógicas:
```
src/
├── data/
│   └── dataset_builder.py      # Capa de construcción e integración de datos crudos
├── features/
│   └── pipeline.py             # Capa de preprocesamiento
├── models/
│   ├── base.py                 # Interfaz abstracta de modelado
│   ├── supervised.py           # Algoritmos predictivos (Red Neuronal, Regresión)
│   └── unsupervised.py         # Algoritmos descriptivos (PAM Clustering)
├── views/
│   └── reports.py              # Capa de presentación (UI/Consola)
├── pipeline_controller.py      # Controlador principal
└── demo.py                     # Simulador basado en los modelos
```

## 3. Patrones de Diseño
* Modelo-Vista-Controlador (MVC): Patrón arquitectónico de alto nivel que estructura todo el directorio.
    * Modelo: Directorios /data, /features y /models. Contienen la lógica matemática, los datos y los algoritmos.
    * Vista: Directorio /views. Se encarga exclusivamente de formatear y mostrar la información al usuario.
    * Controlador: Archivos pipeline_controller.py y demo.py. Dirigen el flujo de información entre el Modelo y la Vista sin realizar cálculos matemáticos.
* Strategy (Estrategia): Permite encapsular una familia de algoritmos y hacerlos intercambiables sin alterar el código cliente.
    * Interfaz: models/base.py
    * Estrategias concretas: models/supervised.py, models/unsupervised.py.
* Pipeline (Tubería): Secuencia lógica y estandarizada para la transformación de datos.
    * Archivos: features/pipeline.py.
* Builder (Constructor): Separa la construcción de un objeto complejo (la muestra de dataset de entrada) de su representación.
    * Archivos: data/dataset_builder.py.
    
## 4. Descripción de los archivos
### Capa de Datos (/data)
* `dataset_builder.py`: Implementa la clase ConstructorDatasetHistorico. Es el responsable de leer la base de datos cruda y los catálogos del gobierno, realizar los cruces relacionales (Left Joins) para desnormalizar el esquema de estrella, aplicar el muestreo estratificado para optimizar el cómputo, y exportar el dataset limpio listo para aplicarle minería de datos.

### Capa de Preprocesamiento (/features)
* `pipeline.py`: Contiene la clase RiesgoCreditoPipeline. Estandariza la imputación de nulos, el escalado robusto para variables financieras y la codificación One-Hot para variables categóricas. Posee una separación estricta entre aprender parámetros estadísticos (fit_transform) y aplicarlos a clientes futuros (transform).

### Capa de Algoritmos (/models)
* `base.py`: Define ModeloEstrategia, una Clase Base Abstracta (ABC). Establece el contrato que todos los algoritmos deben cumplir (entrenar, predecir, guardar, cargar), asegurando el polimorfismo.
* `supervised.py`: Implementa las estrategias predictivas. Destaca la clase RedNeuronalEstrategia, la cual encapsula toda la complejidad de PyTorch (arquitectura MLP, tensores, DataLoaders, funciones de pérdida para clases desbalanceadas y el ciclo explícito de épocas). También incluye una RegresionLogisticaEstrategia como modelo base.
* `unsupervised.py`: Implementa PamClusteringEstrategia. Utiliza el algoritmo K-Medoids y la Distancia de Gower. Guarda la matriz de clientes en mora como referencia, permitiendo calcular la topología y evaluar perfiles en tiempo real.
### Capa de Presentación (/views)
`reports.py`: Es un módulo de funciones de impresión. Generar reportes estadísticos, matrices de confusión interpretadas y tableros de resultados visualmente limpios en la consola.

### Controladores (Raíz de /src)
* `pipeline_controller.py`: Dirige todo el proceso de generación de datos y entrenamiento. Verifica si los datos existen (llamando al Builder si es necesario), realiza la partición estricta en Entrenamiento (64%), Validación (16%) y Prueba (20%), entrena simultáneamente la Red Neuronal y el modelo PAM, y guarda los artefactos generados en disco.
* `demo.py`: Es el simulador de producción interactivo. RCarga los modelos pre-entrenados, despliega un menú para capturar los datos de un cliente y emite un dictamen predictivo. Si el cliente es riesgoso, activa automáticamente el modelo no supervisado para diagnosticar su perfil de morosidad.

## 5. Archivos generados y función
Al ejecutar exitosamente el `pipeline_controller.py`, el sistema genera artefactos dinámicos en los directorios `data/` y `models/`. Estos archivos guardan los datos generados y utlizados por el sistema:
* `/data/Datos_Procesados/dataset_hipotecario_limpio.csv`
Descripción: La base de datos tabular desnormalizada, sin ruido y estratificada.
Función futura: Sirve como el dataset fuente de los entrenamientos.
* `/models/pipeline_preprocesamiento.joblib`
Descripción: Objeto serializado que contiene el estado interno del imputador, el escalador y el codificador del modelo supervisado.
Función: Sirve para predecir el riesgo crediticio de un cliente con las mismas medias y se le divida por las mismas varianzas que a los clientes históricos.
* `/models/modelo_red_neuronal.pth`
Descripción: Archivo binario de PyTorch que guarda exclusivamente los pesos y sesgos (tensores) aprendidos por la red multicapa.
Función futura: Permite realizar la función predictiva del banco rapidamente durante la etapa de producción, sin necesidad de reentrenar.
* `/models/modelo_pam_clustering.joblib`
Descripción: Archivo serializado que contiene el algoritmo K-Medoids entrenado, junto con la matriz histórica de los cohertes riesgosos.
Función futura: Al pasar como entrada un perfil de cŕedito, utiliza la matriz histórica interna para calcular la Distancia de Gower contra la base de referencia y encontrar qué medoide es su "vecino" más cercano.
