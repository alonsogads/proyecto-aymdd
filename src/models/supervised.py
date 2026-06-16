import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression

# Importamos la interfaz que definimos previamente
from .base import ModeloEstrategia

class RegresionLogisticaEstrategia(ModeloEstrategia):
    """
    Estrategia concreta para el modelo de Regresion Logistica.
    Sirve como linea base (baseline) para el problema de clasificacion binaria.
    """
    def __init__(self, random_state=42):
        # Configuramos class_weight='balanced' para manejar el desbalance de clases nativamente
        self.modelo = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state)
        self.entrenado = False

    def entrenar(self, X, y):
        print("Iniciando entrenamiento de Regresion Logistica...")
        self.modelo.fit(X, y)
        self.entrenado = True
        print("Entrenamiento completado.")

    def predecir(self, X):
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
        return self.modelo.predict(X)

    def guardar(self, ruta_archivo):
        if not self.entrenado:
            raise ValueError("No se puede guardar un modelo no entrenado.")
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        joblib.dump(self.modelo, ruta_archivo)
        print(f"Modelo guardado en: {ruta_archivo}")

    @classmethod
    def cargar(cls, ruta_archivo, **kwargs):
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"Archivo no encontrado: {ruta_archivo}")
        estrategia = cls()
        estrategia.modelo = joblib.load(ruta_archivo)
        estrategia.entrenado = True
        print(f"Modelo cargado desde: {ruta_archivo}")
        return estrategia


# === ARQUITECTURA DE LA RED NEURONAL (USO INTERNO) ===

class _ArquitecturaMLP(nn.Module):
    """
    Clase privada que define exclusivamente la arquitectura matematica (capas) de la red.
    Hereda de nn.Module de PyTorch.
    """
    def __init__(self, input_dim):
        super(_ArquitecturaMLP, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 1) # Salida cruda (Logit) para clasificacion binaria
        )

    def forward(self, x):
        return self.red(x)


# === ESTRATEGIA DE LA RED NEURONAL (INTERFAZ PARA EL CONTROLADOR) ===
class RedNeuronalEstrategia(ModeloEstrategia):
    """
    Estrategia concreta que encapsula toda la complejidad de PyTorch (DataLoaders,
    Tensores, Optimizadores, Loss Functions y el ciclo explícito de épocas)
    detras de los metodos simples de la interfaz ModeloEstrategia.
    """
    def __init__(self, epocas=50, batch_size=256, learning_rate=0.001):
        self.epocas = epocas
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.modelo = None # Se instanciara dinamicamente cuando conozcamos la dimension de X
        self.entrenado = False

    def entrenar(self, X, y, X_val=None, y_val=None):
        print("Preparando entorno PyTorch para entrenamiento de Red Neuronal Multicapa...")
        
        input_dim = X.shape[1]
        self.modelo = _ArquitecturaMLP(input_dim)
        
        # Tensores de Entrenamiento
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # Tensores de Validacion (si se proveen)
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        num_negativos = (y == 0).sum()
        num_positivos = (y == 1).sum()
        peso_clase_positiva = torch.tensor([num_negativos / num_positivos], dtype=torch.float32)
        
        criterio = nn.BCEWithLogitsLoss(pos_weight=peso_clase_positiva)
        optimizador = optim.Adam(self.modelo.parameters(), lr=self.learning_rate)
        
        for epoca in range(self.epocas):
            self.modelo.train()
            perdida_acumulada = 0.0
            
            for lote_X, lote_y in dataloader:
                optimizador.zero_grad()
                predicciones_crudas = self.modelo(lote_X)
                perdida = criterio(predicciones_crudas, lote_y)
                perdida.backward()
                optimizador.step()
                perdida_acumulada += perdida.item()
                
            perdida_promedio = perdida_acumulada / len(dataloader)
            
            # Si tenemos datos de validacion, calculamos la perdida de validacion
            if X_val is not None and y_val is not None:
                self.modelo.eval()
                with torch.no_grad():
                    val_predicciones = self.modelo(X_val_tensor)
                    val_perdida = criterio(val_predicciones, y_val_tensor).item()
                    
                # Imprimimos progreso cada 10 epocas o en la ultima epoca
                if (epoca + 1) % 10 == 0 or (epoca + 1) == self.epocas:
                    print(f" -> Epoca [{epoca+1}/{self.epocas}] - Loss Entrenamiento: {perdida_promedio:.4f} | Loss Validacion: {val_perdida:.4f}")
            else:
                if (epoca + 1) % 10 == 0 or (epoca + 1) == self.epocas:
                    print(f" -> Epoca [{epoca+1}/{self.epocas}] - Loss Entrenamiento: {perdida_promedio:.4f}")
                
        self.entrenado = True
        print("Entrenamiento de Red Neuronal completado.")

    def predecir(self, X):
        if not self.entrenado or self.modelo is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
            
        # Transformamos la entrada a tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Ponemos el modelo en modo evaluacion (desactiva Dropout para consistencia)
        self.modelo.eval()
        
        # Apagamos el calculo de gradientes para ahorrar memoria e ir mas rapido
        with torch.no_grad():
            predicciones_crudas = self.modelo(X_tensor)
            # Aplicamos funcion sigmoide para transformar logits a probabilidades (0 a 1)
            probabilidades = torch.sigmoid(predicciones_crudas)
            # Clasificamos con el umbral matematico estandar de 0.5
            predicciones_binarias = (probabilidades >= 0.5).int().numpy().flatten()
            
        return predicciones_binarias

    def guardar(self, ruta_archivo):
        if not self.entrenado or self.modelo is None:
            raise ValueError("No se puede guardar un modelo no entrenado.")
        
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        # En PyTorch se recomienda guardar el diccionario de estados (state_dict) en lugar del objeto completo
        torch.save(self.modelo.state_dict(), ruta_archivo)
        print(f"Pesos de la Red Neuronal guardados en: {ruta_archivo}")

    @classmethod
    def cargar(cls, ruta_archivo, **kwargs):
        """
        Para cargar un modelo de PyTorch, necesitamos conocer la dimension de entrada (input_dim)
        para poder reconstruir la arquitectura antes de inyectarle los pesos guardados.
        """
        if 'input_dim' not in kwargs:
            raise ValueError("Debe proporcionar 'input_dim' (numero de caracteristicas) en kwargs para reconstruir la red.")
            
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"Archivo no encontrado: {ruta_archivo}")
            
        input_dim = kwargs['input_dim']
        
        # Instanciamos la estrategia vacia
        estrategia = cls()
        
        # Reconstruimos la arquitectura matematica
        estrategia.modelo = _ArquitecturaMLP(input_dim)
        
        # Inyectamos los pesos (state_dict) desde el disco duro
        estrategia.modelo.load_state_dict(torch.load(ruta_archivo))
        
        # Forzamos el modo evaluacion como medida de seguridad
        estrategia.modelo.eval()
        estrategia.entrenado = True
        print(f"Pesos de la Red Neuronal cargados desde: {ruta_archivo}")
        return estrategia