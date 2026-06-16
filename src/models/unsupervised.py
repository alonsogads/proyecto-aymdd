import os
import joblib
import numpy as np
import pandas as pd
import gower
from sklearn_extra.cluster import KMedoids

# Importamos la interfaz base
from .base import ModeloEstrategia

class PamClusteringEstrategia(ModeloEstrategia):
    """
    Estrategia concreta para el modelo de agrupamiento K-Medoids (PAM).
    Utiliza la distancia de Gower para manejar datos financieros mixtos.
    """
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Instanciamos el modelo indicando que recibira una matriz de distancias ya calculada
        self.modelo = KMedoids(
            n_clusters=self.n_clusters, 
            metric='precomputed', 
            method='pam', 
            init='heuristic', 
            random_state=self.random_state
        )
        
        self.entrenado = False
        self.X_train_referencia = None
        self.indices_medoides = None

    def entrenar(self, X, y=None):
        """
        Calcula la matriz de Gower y encuentra los perfiles representativos.
        
        Nota Arquitectonica: A diferencia de la Red Neuronal que recibe el output del 
        Pipeline (variables One-Hot), esta estrategia espera un DataFrame de Pandas 
        con sus variables categoricas intactas para que Gower funcione correctamente.
        """
        print("Iniciando entrenamiento de clustering PAM con Distancia de Gower...")
        
        # Guardamos una copia de los datos de entrenamiento.
        # Esto es vital para poder calcular las distancias de clientes futuros.
        self.X_train_referencia = X.copy()
        
        # 1. Calculamos la matriz de distancias cruzadas
        print("Calculando matriz de distancias de Gower (Esto puede tomar tiempo)...")
        matriz_distancias = gower.gower_matrix(self.X_train_referencia)
        
        # 2. Ajustamos el modelo para que encuentre los medoides
        print("Ajustando particiones y buscando clientes representativos...")
        self.modelo.fit(matriz_distancias)
        
        self.indices_medoides = self.modelo.medoid_indices_
        self.entrenado = True
        
        print(f"Entrenamiento completado. Medoides centrales en los indices: {self.indices_medoides}")

    def predecir(self, X):
        """
        Asigna uno de los clusters existentes a nuevos registros.
        """
        if not self.entrenado or self.X_train_referencia is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir o asignar clusters.")
            
        print("Calculando distancias de los nuevos registros contra la base de referencia...")
        
        # Calculamos la distancia de los registros NUEVOS (X) contra los que ya conoce (X_train_referencia)
        # Esto genera una matriz asimetrica que el metodo predict de KMedoids sabe interpretar
        matriz_distancias_nuevas = gower.gower_matrix(data_x=X, data_y=self.X_train_referencia)
        
        # Asignamos la etiqueta (0 o 1) segun el medoide mas cercano
        predicciones_cluster = self.modelo.predict(matriz_distancias_nuevas)
        
        return predicciones_cluster

    def guardar(self, ruta_archivo):
        """
        Guarda el modelo y el dataframe de referencia necesario para inferencia.
        """
        if not self.entrenado:
            raise ValueError("No se puede guardar un modelo que no ha sido entrenado.")
            
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        
        # Empaquetamos todo lo necesario en un solo diccionario (artefacto)
        artefacto_produccion = {
            'modelo_pam': self.modelo,
            'X_train_referencia': self.X_train_referencia,
            'indices_medoides': self.indices_medoides
        }
        
        joblib.dump(artefacto_produccion, ruta_archivo)
        print(f"Modelo PAM (con datos de referencia) guardado exitosamente en: {ruta_archivo}")

    @classmethod
    def cargar(cls, ruta_archivo, **kwargs):
        """
        Reconstruye la estrategia cargando el modelo y la matriz de referencia.
        """
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"No se encontro el archivo en: {ruta_archivo}")
            
        artefacto_cargado = joblib.load(ruta_archivo)
        
        # Creamos una instancia vacia
        estrategia = cls()
        
        # Inyectamos los componentes guardados
        estrategia.modelo = artefacto_cargado['modelo_pam']
        estrategia.X_train_referencia = artefacto_cargado['X_train_referencia']
        estrategia.indices_medoides = artefacto_cargado['indices_medoides']
        estrategia.entrenado = True
        
        print(f"Modelo PAM cargado exitosamente desde: {ruta_archivo}")
        return estrategia