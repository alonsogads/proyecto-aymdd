# Proyecto Final - Minería de datos aplicada a México

## Descripción general
El proyecto integrado tiene como objetivo que cada quién aplique, de forma cohesionada, los principales temas del curso sobre un problema real y relevante de México. La solución deberá cubri todas las estapas del ciclo de vida d eun proyecto de minería de datos: definición del problema, obtención y exploración de datos, modelado supervisado y no supervisado, implementación técnica con buenas prácticas de ingenería de software, y comunicación formal de resultados.
## Tema, problema y justificación
Cada proyecto seleccionará libremente su tema de una problemática en la sociedad mexicana de alguna de las siguientes áreas (no expluyentes):
* Seguridad pública
* Salud pública
* Educación
* Medio ambiente
* Agricultura y alimentación
* Economía y finanzas
* Transporte y movilidad
* Turismo y cultura
* Vivienda y urbanismo
* Energía
* Comercio e industria
La justificación del tema deberá responder explícitamente a las siguientes cuatro preguntas:
1. ¿Cuál es el problema? Descripción concreta y acotada del fenómeno que se analiza.
2. ¿Por qué es relevante? Impacto social, económico o ambiental; evidencia de que el problema existe y es significativo.
3. ¿Qué se busca lograr? Objetivo claro y medible del análisis (por ejemplo, predecir, segmentar, identificar patrones).
4. ¿Por qué este dataset? Pertinencia de los datos elegidos para responder el objetivo planteado.
## Datos (dataset)
Requisitos mínimos obligatorios:
* Datos reales relacionados con México (no generados artificalmente).
* Mínimo 100,000 registros (filas) y al menos 8 variables relevantes (columnas)
* Presencia de variables numéricas y catégoricas.
* Fuente verificable, pública y citable (URL directa o DOI).
* Los datos deben permitir tanto modelado supervisado como no supervisado.
Restricciones: No se acerptarán datasets generados artificalmente, aumentados por duplicación o síntesis de registros, ni datasets de uso privado sin fuente verficable. Los datos deben poder ser descargados o accedidos por el equipo docente para verificación.
Fuentes sugeridas: Se recomienda utilizar las siguientes fuentes con datos abiertos verificables y descargables:
* INEGI: https://www.gob.mx/salud/documentos/datos-abiertos-152127
* Datos abiertos: https://www.inegi.org.mx/datosabiertos/
* Plataforma Nacional de Datos Abiertos: google.com/url?sa=D&q=https://datos.gob.mx&ust=1781053020000000&usg=AOvVaw0Pof9yXKNM2RcOp7wBJ5wS&hl=es
* CONEVAL: https://www.coneval.org.mx/Medicion/Paginas/Programas_BD_municipal.aspx
* Dirección General de Epidemiología: https://www.gob.mx/salud/documentos/datos-abiertos-152127
* SESNSP: https://www.gob.mx/sesnsp/acciones-y-programas/datos-abiertos-de-incidencia-delictiva
* CONAGUA: https://datos.conagua.gob.mx/views/index_datos_abiertos.html
* Sistema de Información Hidrológica (SIH): https://sih.conagua.gob.mx/
* DENUE: https://www.inegi.org.mx/app/mapa/denue/
* API Banxico: https://www.banxico.org.mx/SieAPIRest/service/v1/
* Datos Abiertos CDMX: google.com/url?sa=D&q=https://datos.cdmx.gob.mx&ust=1781053020000000&usg=AOvVaw1XvBpwIOe0VfJcRTXLc9Kj&hl=es
## Análisis Exploratorio de Datos (EDA)
El EDA es la primera etapa analítica y deberá realizarse antes de cualquier modelado. Su propósito es comprender la estructura, calidad y distribución de los datos, e informar las decisiones de preprocesamiento y modelado posteriores. El EDA deberá incluir, como mínimo:
1. Descripción general: dimensiones del dataser (filas x columnas), tipos de datos por variable, fuente y fecha de extracción.
2. Calidad de datos: conteo de valores faltantes por columna (y porcentaje), registros duplicados, y estrategia de manejo de  faltantes elegida con justificación.
3. Estadísticas descriptivas: media, mediana, desviación estándar, cuartiles, y rango para variables numéricas; frecuencias y moda para variables categóricas.
4. Detección de valores atípicos: uso de boxplots, regla IQR o z-score, decisión justificada sobre su tratamiento.
5. Distribuciones: histogramas de todas las variables numéricas relevantes con interpretación.
6. Relaciones entre variables: matriz de correlaciones (Pearson o Spearman) con heatmap, scatter plots de pares relevantes.
7. Análisis de variables categóricas: gráficas de barras con frecuencias; análisis de balance de clases si aplica al modelado supervisado.
8. Visualizaciones adicionales (si apica): mapas geográficos, series de tiempo y otras gráficas pertienentes al dominio del problema.
Importante: Cada visualización debe ir acompañada de una interpretación escrita que explique qué se observa y qué implicación tiene para el análisis. No basta con mostrar la gráfica.
## Modelo supervisado
### Qué se debe construir
Deberán entrenar al menos un modelo de aprendizaje supervizado. el tipo de tarea (clasificación o regresión) deberá derivarse directamente del objetivo proyecto.
Algoritmos sugeridos (no exclusivos): Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Gradient Boosting, XGBoost, Support Vector Machine, Linear/Ridge/Lasso Regression.
### Proceso requerido
1. Definición de la variable objetivo: indicar qué se predice y por qué esa variable es la más adecuada para el objetivo del proyecto.
2. Preprocesamiento específico para el modelo: codificación de variables catégoricas, normalización o estandarización de variables numéricas, manejo de desbalance de clases si aplica.
3. Selección de variables de entrada; justificar qué variables se incluyen y cuáles se excluyen (con base en el EDA, dominio dle problema o técnicas de selección de características).
4. División de datos: separación en conjunto de entrenamiento y prueba (mínimo 70/30 o 80/20), con semilla aleatoria fija para reproductibilidad.
5. Entrenamiento y ajuste: ajuste de hiperparámetros con validación cruzada o búsqueda en cuadrícula, documentando la configuración final.
6. Evaluación e interpretación: métricas completas según la tarea (explicadas abajo) e interpretación de resultados en el contexto del problema real.
7. Comparación con linea base: el modelo debe superar un clasificador/regreso trivial (por ejemplo, clasificador mayoritario o media del objetivo); de no superar se deberá analizar el por qué.
### Métricas de evaluación
| Tarea | Méticas requeridas |
---
| Clasificación | Accuracy, Precision, Recall, F1-score (macro y ponderado), matriz de confusión, curva ROC y AUC si aplica. |
| Regresión | MAE, MSE, RMSE, R2 (coeficiente de determinación), gráfica de residuales. |
### Persistencia del modelo
El modelo entrenado deberá guardarse en disco usando `pickle` o `joblib` y cargarse correctamente en una demostración separada del entrenamiento. La persistencia es un requisito técnico evaluado.
## Agrupamiento (Clustering)
### Qué se debe construir
Deberán aplicar obligatoriamente al menos una técnica de agrupamiento no supervisado. El agrupamiento debe tener un propósito analítico claro: identificar segmentos, perfiles o patrones ocultos en los datos que complementen o enriquezcan los hallazgos del modelo supervisado.
Algoritmos sugeridos: K-Means, DBSCAN, Agrupamiento jerárquico (Ward, complete, average), Gaussian Mixture Models.
### Proceso requerido
1. Selección y justificación de variables: indicar qué variables se usan para el agrupamiento y por qué (pueden diferrir de las del modelo supervisado).
2. Preprocesamiento: normalización de variables (obligatorio para K-Means y KDE); reducción de dimensionalidad con PCA si se trabaja con muchas variables.
3. Selección del algoritmo y justificación: por qué ese algoritmo es adecuado para los datos (considerar forma de los grupos, escala, ruido).
4. Elección del número de grupos (si el algoritmo lo requiere): uso de al menos uno de los siguientes criterios: método del codo, coeficiente de silueta, dendrograma.
5. Visualización de grupos: scatter plot con colores por cluster (con PCA o t-SNE) para reducción si hay más de 2 dimensiones).
6. Interpretación de perfiles: descripción de cada grupo con base en las variables originales (estadísticas por cluster); los grupos deben tener un nombre o etiqueta interpretativa, no solo un número.
## Requerimientos técnicos
### Lenguaje y librerías
Python es el lenguaje obligatorio. Se recomienda el uso de librerias que hemos visto en el curso, tales como: pandas, numpy, scikit-lear, matplotlib, seaborn, plotly (para visualización interactivas en Quarto).
### Programación orientada a objetos (POO)
El código fuente deberá estar organizao usando clases y objetos. No se acepta un único script lineal o un notebook sin estructura. Se evaluará:
* Encapsulamiento de responsabilidades en clases coherentes.
* Uso de herencia o composición donde sea natural.
* Separación entre lógica de datos, modelado y reporte.
### Patrón de diseño
Deberá implementarse al menos un patrón de diseño de software, elegido entre los siguientes, y justificar su elección en el reporte.
| Patrón | Cuándo es adecuado |
|---|---|
| Pipeline | Cuando el procesamiento de datos ocurre en etapas secuenciales bien definidad (ej, limpieza -> encoding -> escalado -> modelo). |
| Factory method | Cuando se necesita crear instancias de diferentes modelos o algoritmos de forma intercambiable sin cambiar el código cliente |
| Strategy | Cuando se desea intercambiar algoritmos de preprocesamiento, modelado o evaluación en tiempo de ejecución |
| Repository | Cuando se desea abstraer el acceso a los datos (CSV, base de datos, API) detrás de una interfaz uniforme |
### Reproductibilidad y control de versiones
* Todas las semillas aleatorias (random_state) deben fijarse explcítiamente.
* El repositorio de GitHub debe tener historial de commits que evidencie el progreso continuo.
* Archivo requirements.txt con todas las dependencias y sus versiones exactas (generado con pip freeze o equivalente).
* Archivo .gitignore adecaudo (excluir __pycache__, .ipynb_checkpoints, archivos grandes, etc.)
## Documentación con Quarto y GitHub Pages
El proyecto deberá incluir documentación técnica interactiva generado con Quarto (.qmd), compilada como sitio HTML estático en la carpeta docs/ y desplegada públicamente con GitHub Pages.
### Contenido mínimo del sitio
1. Descripción del problema y del dataset
2. EDA con visualizaciones (estaícas o interactivas con Plotly).
3. Resultados del modelo supervisado (métricas y gráficas).
4. Resultados del clustering (visualización e interpretación de grupos).
5. Instrucciones de instalación y ejecución del proyecto.
### Configuración mínima
* Archivo _quarto.yml en la raíz del repositorio con output_dir: docs
* GitHub Pages configurado en : Settings -> Pages -> Branch: main/docs
* URL del sitio desplegado incluida en el README.md
## Entregables
1.  Dataset o enlace: Datos en data/ o enlace directo a fuente oficial en
el README.md
2. Notebooks: Notebooks organizados en notebooks/ (EDA, modelado, clustering)
3. Código fuente: Clases y módulos Python en src/
4. Modelos serializados: Archivos .pkl o .joblib en models/
5. Documentación Quarto: Sitio HTML en docs/ desplegado en GitHub Pages
6. Reporte final: PDF en reports/ (de preferencia en Latex)
7. Diagrama UML: Diagrama de arquitectura en diagrams/
8. Presentación (slides): Archivo de slides en reports/
9. README.md: Instrucciones de instalación, ejecución y URL del sitio Quarto
10. requirements.txt: Dependencias con versiones exactas
### Estructura sugerida del repositorio
```
Proyecto_Mineria_Datos/
|-- data/ # Datos crudos y procesados (o enlace a fuente)
|-- notebooks/ # Notebooks de EDA, modelado y clustering
|-- src/ # Código fuente Python (clases, módulos)
|-- models/ # Modelos serializados (.pkl / .joblib)
|-- reports/ # Reporte PDF y slides de presentación
|-- diagrams/ # Diagrama UML de la arquitectura
|-- docs/ # Sitio Quarto compilado (GitHub Pages)
|-- _quarto.yml # Configuración de Quarto
|-- README.md
|-- requirements.txt
└-- .gitignore
```
## Reporte final
El reporte es un documento técnico formal en PDF. Deberá ser autocontenido: un lector que no haya visto el código ni los datos debe poder entender completamente el trabajo realizado.
Secciones obligatorias:
1. Portada: nombre del proyecto, datos del alumno, fecha.
2. Introducción: Problema, objetivo, justificación y valor del proyecto.
3. Descripción del dataset: fuente, variables (con definición de cada una), tamaño, y limitaciones conocidas de los datos.
4. Limpieza y preparación de datos: decisiones tomadas sobre valores faltantes, duplicados, outliers y transformaciones; todo con justificación.
5. Análisis Exploratorio: visualizaciones con interpretación narrativa.
6. Modelo supervisado: descripción del proceso, resultados, métricas y análisis crítico de los resultados.
7. Clustering: descripción del proceso, criterio de selección del número de grupos, visualización e interpretación de perfiles.
8. Arquitectura y diseño: patrón de diseño implementado con justificación; diagrama UML de la arquitectura.
9. Reutilización del modelo: demostrar que el modelo guardado puede cargarse y usarse para hacer predicciones nuevas sin re-entrenar.
10. Conclusiones: hallazgos principales, limitaciones del análisis y trabajo futuro.
11. Referencias: formato APA o IEEE, mínimo 5 referencias (incluyendo fuente del dataset y literatura técnica).
## Presentación oral
* Duración máxima: 15 minutos + preguntas del equipo docente
* Se realizará una demostración en vivo: cargar el modelo guardado y ejecutar una predicción nueva en tiempo real.
Temas que deberá cubrir la presentación.
1. Problema y dataset (motivación, fuente, características).
2. Hallazgos del EDA (máximo 3 visualizaciones clave con interpretación).
3. Modelo supervisado: proceso, métricas y conclusión sobre su desempeño.
4. Clustering: algoritmo elegido, número de grupos y descripción de cada perfil.
5. Patrón de diseño: cuál, porqué y cómo se implementó.
6. Demostración en vivo de la reutilizción dle modelo.
7. Conclusiones y limitaciones.
## Causas de penalización
1. Uso de dataset generado artificalmente
2. Código no reproducible, sin semillas fijas o sin requirements.txt
3. Modelo guardado que no pueda cargarse y ejecutarse independientemente del notebook de entrenamiento.
4. Entregables subidos después de la presentación oral.