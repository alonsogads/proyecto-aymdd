from abc import ABC, abstractmethod

class ModeloEstrategia(ABC):
    """
    Clase Base Abstracta que define el contrato para todos los modelos del sistema.
    Implementa el patron Strategy para permitir el intercambio de algoritmos
    (Supervisados o No Supervisados) de forma explicita para el controlador.
    """

    @abstractmethod
    def entrenar(self, X, y=None):
        """
        Ejecuta el ciclo de aprendizaje del algoritmo.
        
        Parametros:
        - X: Matriz de caracteristicas (numpy array o tensor).
        - y: Vector de etiquetas (opcional, ya que los modelos no supervisados no lo usan).
        """
        pass

    @abstractmethod
    def predecir(self, X):
        """
        Toma una matriz de datos nuevos y devuelve las predicciones o las
        etiquetas de los clusteres asignados.
        
        Parametros:
        - X: Matriz de caracteristicas a evaluar.
        
        Retorna:
        - Predicciones generadas por el modelo entrenado.
        """
        pass

    @abstractmethod
    def guardar(self, ruta_archivo):
        """
        Persiste los pesos, centroides o parametros del modelo en el disco duro.
        Cada algoritmo (ej. PyTorch vs Scikit-Learn) definira como guardarse a si mismo.
        
        Parametros:
        - ruta_archivo: Ruta del sistema operativo donde se guardara el artefacto.
        """
        pass

    @classmethod
    @abstractmethod
    def cargar(cls, ruta_archivo, **kwargs):
        """
        Carga un modelo previamente guardado en disco y lo devuelve listo para inferencia.
        Es un metodo de clase (classmethod) porque se llama antes de tener una instancia viva.
        
        Parametros:
        - ruta_archivo: Ruta del sistema operativo desde donde se leera el artefacto.
        - kwargs: Argumentos adicionales (como dimensiones de entrada para redes neuronales).
        """
        pass