# Proyecto Final: Minería de Datos aplicada a México
Repositorio del Proyecto: **Riesgo y Deterioro de la Cartera de la Vivienda**

Universidad Nacional Autónoma de México
Facultad de Ciencias
Almacenes y Minería de Datos

* **Alumno:** Gustavo Alonso Domínguez Sánchez - 418047121

* **Profesor:** Jessica Santizo Galicia
* **Ayudante:** Diego Antonio Villalba González
* **Ayudante Lab:** Ares Gael Castro Romero


## Descripción del Proyecto

Este repositorio contiene el desarrollo integral de un sistema de Minería de Datos orientado a la evaluación del riesgo en el portafolio hipotecario mexicano. Utilizando la [Base Histórica de la Comisión Nacional Bancaria y de Valores (CNBV)](https://portafolioinfo.cnbv.gob.mx/Paginas/Inicio.aspx). Este proyecto aplica metodologías formales de ciencia de datos para anticipar y perfilar el deterioro crediticio.

El proyecto resuelve la problemática del riesgo a través de dos enfoques metodológicos:
1. Objetivo Predictivo (Aprendizaje Supervisado): Implementación de Redes Neuronales Profundas (PyTorch) para clasificar la probabilidad de incumplimiento de nuevas cohortes de crédito, resolviendo el desbalance de clases mediante funciones de pérdida ponderadas.
2. Objetivo Descriptivo (Aprendizaje No Supervisado): Implementación de algoritmos de agrupamiento avanzados (K-Medoids) para segmentar la cartera vencida y aislar perfiles estructurales de morosidad.

## Documentación Interactiva

Toda la fundamentación matemática, el Análisis Exploratorio de Datos (EDA) exhaustivo y las decisiones de diseño metodológico se encuentran publicados y accesibles de manera interactiva.

Sitio oficial del proyecto:
[Documentación Interactiva (Quarto Pages)](https://alonsogads.github.io/proyecto-aymdd/)

## Arquitectura y Estructura del Repositorio

El proyecto abandona la dependencia exclusiva de cuadernos de experimentación para adoptar una arquitectura de software Orientada a Objetos (POO), implementando patrones de diseño corporativos (Modelo-Vista-Controlador, Strategy, Builder y Pipeline) que garantizan su viabilidad como sistema de producción MLOps.

La estructura de directorios es la siguiente:

- data/: Tabla de hechos crudo (instrucciones de descarga) y catálogos dimensionales.
- docs/: Sitio Quarto compilado (GitHub Pages)
- models/: Artefactos y modelos serializados generados por el sistema (.pth para PyTorch, .joblib para Scikit-Learn).
- notebooks/: Jupyter Notebooks de EDA, modelado y clustering.
- reports/: Documentación técnica en formato LaTeX y PDF (Reporte Final).
- src/: Código fuente de producción para la ingesta, entrenamiento e inferencia en tiempo real.
- diagrams/: Diagrama UML de la arquitectura del código de producción

## Configuración del Entorno Virtual (Environment)

Para asegurar la reproducibilidad del sistema y evitar conflictos entre librerías (especialmente con PyTorch y los cálculos de Gower), es estrictamente necesario ejecutar el código dentro de un entorno virtual aislado.

Siga estas instrucciones desde la terminal en el directorio raíz del proyecto:

1. Creación del entorno virtual:
```shell
python3 -m venv venv
```
2. Activación del entorno virtual:
- En sistemas operativos Linux / macOS:
```shell
source venv/bin/activate
```
- En sistemas operativos Windows (Símbolo del sistema o PowerShell):
```shell
  venv\Scripts\activate
```
3. Instalación de dependencias:
Una vez que el entorno virtual se encuentre activo, instale los requerimientos ejecutando:
```shell
pip install -r requirements.txt
```
## Instrucciones de Ejecución

Todo el código operativo de producción reside en la carpeta `/src`. Asegúrese de tener el entorno virtual activado y navegue a dicho directorio:
```shell
cd src
```
1. Control y Entrenamiento:
Para ejecutar la construcción de la base de datos histórica, el preprocesamiento matemático y el entrenamiento algorítmico completo (tanto de la Red Neuronal como del algoritmo PAM), ejecute el controlador principal:
```shell
python3 pipeline_controller.py
```
El sistema operará de manera autónoma, particionará los datos, validará la arquitectura e imprimirá en consola las métricas de rendimiento y la matriz de confusión sobre el conjunto de prueba. Al finalizar, persistirá los modelos entrenados en la carpeta `/models/`.

2. Inferencia Continua (Simulador de Originación)
Una vez generados los artefactos en el paso anterior, puede desplegar el sistema de evaluación interactivo. Este módulo simula la interfaz para un análisis de crédito para emitir un dictamen en tiempo real.
```shell
python3 demo.py
```
El sistema le solicitará capturar atributos socioeconómicos y financieros utilizando los catálogos oficiales. Posteriormente, la información pasará por un prroceso continuo: el modelo predictivo evaluará el riesgo y, en caso de detectar un deterioro significativo, el modelo descriptivo categorizará a la cohorte dentro de un perfil específico de morosidad.