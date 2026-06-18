---
title: "Proyecto Final: Riesgo y Deterioro de la Cartera de la Vivienda"
author: "Gustavo Alonso Dominguez Sanchez"
institute: "Facultad de Ciencias, UNAM"
format: 
  html:
    toc: false
    page-layout: full
---

Esta página web consolida el análisis, diseño y desarrollo de un sistema integral de **Minería de Datos** orientado a la evaluación del riesgo en el portafolio hipotecario mexicano, utilizando la Base Histórica auditada por la Comisión Nacional Bancaria y de Valores (CNBV).

### Estructura del Proyecto

La investigación está dividida en tres fases principales:

1. **[Análisis Exploratorio de Datos (EDA)](notebooks/EDA.ipynb)**
   * Integración de dimensiones (esquema de estrella) y muestreo estratificado.
   * Análisis univariado y bivariado para perfilar las características demográficas y de originación de las cohortes.
   * Justificación estadística para la selección de características y tratamiento de valores atípicos.

2. **[Modelo Supervisado (Redes Neuronales)](notebooks/modelo_supervisado.ipynb)**
   * Desarrollo de un Perceptrón Multicapa (*Feed-Forward*) para la predicción de la probabilidad de incumplimiento.
   * Manejo nativo del desbalance de clases mediante funciones de pérdida ponderadas.
   * Evaluación del desempeño algorítmico priorizando la sensibilidad (*Recall*) sobre la cartera deteriorada.

3. **[Modelo No Supervisado (Clustering PAM)](notebooks/modelo_no_supervisado.ipynb)**
   * Segmentación exclusiva de la cartera en Riesgo Alto para descubrir arquetipos estructurales de morosidad.
   * Implementación del algoritmo *Partitioning Around Medoids* (PAM) con la métrica de Distancia de Gower para matrices de datos mixtos.
   * Proyección dimensional mediante t-SNE y auditoría visual de los medoides.

> *Para consultar el código fuente y el reporte formal en formato PDF, visita el repositorio oficial del proyecto en GitHub.*
