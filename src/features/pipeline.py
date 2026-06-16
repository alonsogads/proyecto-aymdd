import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class RiesgoCreditoPipeline:
    """
    Clase que implementa el patron de diseno Pipeline.
    Su objetivo es encapsular y aislar la logica de preprocesamiento de datos:
    imputacion de valores nulos, codificacion de variables categoricas y
    escalado de variables numericas.
    """
    
    def __init__(self):
        # 1. Definimos listas explicitas con los nombres exactos de nuestras columnas
        self.columnas_numericas = [
            'saldo_insoluto_final_periodo', 
            'tasa_ponderada'
        ]
        
        self.columnas_categoricas = [
            'intervalo_edades', 
            'intervalo_ingreso_acreditado', 
            'sector_laboral', 
            'estado', 
            'destino_credito', 
            'segmento_vivienda', 
            'moneda'
        ]
        
        # 2. Instanciamos las herramientas matematicas necesarias
        # Usamos la mediana para numeros y la moda para categorias en caso de valores vacios
        self.imputador_num = SimpleImputer(strategy='median')
        self.imputador_cat = SimpleImputer(strategy='most_frequent')
        
        # Elegimos RobustScaler para no distorsionar las distancias con los saldos atipicos
        self.escalador = RobustScaler()
        
        # El parametro handle_unknown='ignore' es vital para el entorno de produccion,
        # asi el sistema no colapsa si llega un cliente con una categoria nunca antes vista.
        self.codificador = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # 3. Variables de control interno
        self.entrenado = False
        self.nombres_caracteristicas_finales = []

    def fit_transform(self, df_entrada):
        """
        Aprende los parametros estadisticos (medias, modas, rangos) exclusivamentte 
        del conjunto de entrenamiento (Train) y transforma los datos.
        """
        print("Iniciando fit_transform del Pipeline...")
        
        # Trabajamos sobre una copia para no alterar el DataFrame original
        df = df_entrada.copy()
        
        # --- Procesamiento de Variables Numericas ---
        datos_num = df[self.columnas_numericas].copy()
        
        # Aprendemos y aplicamos la imputacion
        datos_num_imputados = self.imputador_num.fit_transform(datos_num)
        
        # Aprendemos y aplicamos el escalado robusto
        datos_num_escalados = self.escalador.fit_transform(datos_num_imputados)
        
        # --- Procesamiento de Variables Categoricas ---
        datos_cat = df[self.columnas_categoricas].copy()
        
        # Aprendemos y aplicamos la imputacion
        datos_cat_imputados = self.imputador_cat.fit_transform(datos_cat)
        
        # Aprendemos y aplicamos la codificacion (One-Hot)
        datos_cat_codificados = self.codificador.fit_transform(datos_cat_imputados)
        
        # Extraemos los nombres de las nuevas columnas generadas por el codificador
        nombres_cat_generados = self.codificador.get_feature_names_out(self.columnas_categoricas)
        
        # --- Integracion ---
        # Concatenamos de forma explicita los arreglos numericos y categoricos
        matriz_final = np.hstack((datos_num_escalados, datos_cat_codificados))
        
        # Guardamos el orden exacto de las columnas para referencias futuras
        self.nombres_caracteristicas_finales = self.columnas_numericas + list(nombres_cat_generados)
        self.entrenado = True
        
        print(f"Pipeline ajustado exitosamente. Total de caracteristicas finales: {len(self.nombres_caracteristicas_finales)}")
        return matriz_final

    def transform(self, df_entrada):
        """
        Aplica las transformaciones matematicas previamente aprendidas a un nuevo 
        conjunto de datos (Validacion, Prueba o Inferencia en Produccion).
        NUNCA aprende informacion nueva aqui.
        """
        if not self.entrenado:
            raise ValueError("El pipeline no ha sido entrenado. Ejecuta fit_transform primero.")
            
        df = df_entrada.copy()
        
        # --- Transformacion Numerica ---
        datos_num = df[self.columnas_numericas].copy()
        datos_num_imputados = self.imputador_num.transform(datos_num)
        datos_num_escalados = self.escalador.transform(datos_num_imputados)
        
        # --- Transformacion Categorica ---
        datos_cat = df[self.columnas_categoricas].copy()
        datos_cat_imputados = self.imputador_cat.transform(datos_cat)
        datos_cat_codificados = self.codificador.transform(datos_cat_imputados)
        
        # --- Integracion ---
        matriz_final = np.hstack((datos_num_escalados, datos_cat_codificados))
        return matriz_final

    def guardar(self, ruta_archivo):
        """
        Persiste el estado actual del Pipeline en disco para su uso posterior.
        """
        if not self.entrenado:
            raise ValueError("No se puede guardar un pipeline que no ha sido entrenado.")
            
        # Aseguramos que el directorio exista
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        joblib.dump(self, ruta_archivo)
        print(f"Pipeline guardado exitosamente en: {ruta_archivo}")

    @classmethod
    def cargar(cls, ruta_archivo):
        """
        Carga un Pipeline previamente guardado desde el disco.
        """
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"No se encontro el archivo en: {ruta_archivo}")
            
        pipeline_cargado = joblib.load(ruta_archivo)
        print(f"Pipeline cargado exitosamente desde: {ruta_archivo}")
        return pipeline_cargado