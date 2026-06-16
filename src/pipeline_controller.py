import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Importamos nuestras herramientas construidas (El Modelo)
from features.pipeline import RiesgoCreditoPipeline
from models.supervised import RedNeuronalEstrategia, RegresionLogisticaEstrategia

def configurar_argumentos():
    """
    Configura y lee los argumentos enviados desde la terminal.
    Implementa la seleccion dinamica del modelo sin modificar el codigo fuente.
    """
    parser = argparse.ArgumentParser(description="Director de entrenamiento de Modelos de Riesgo")
    
    # Definimos el argumento '--modelo' con opciones estrictas
    parser.add_argument(
        '--modelo',
        type=str,
        choices=['red_neuronal', 'regresion_logistica'],
        default='red_neuronal',
        help="Selecciona el algoritmo a entrenar (opciones: red_neuronal, regresion_logistica. Por defecto: red_neuronal)"
    )
    
    return parser.parse_args()

def principal():
    """
    Funcion principal (Controlador) que orquesta todo el proceso de entrenamiento.
    """
    args = configurar_argumentos()
    
    print("=" * 60)
    print("CONTROLADOR DE ENTRENAMIENTO - RIESGO CREDITICIO")
    print(f"Modelo seleccionado: {args.modelo.upper()}")
    print("=" * 60)

    # --- PASO 1: CARGA DE DATOS ---
    ruta_datos = "../data/Datos_Procesados/dataset_hipotecario_limpio.csv"
    print(f"[1/6] Cargando datos desde: {ruta_datos}")
    
    if not os.path.exists(ruta_datos):
        raise FileNotFoundError("El archivo de datos no existe. Verifica la ruta.")
        
    df = pd.read_csv(ruta_datos)
    df_filtrado = df[df['etapa'].isin(['Etapa 1', 'Etapa 3'])].copy()
    df_filtrado['objetivo_binario'] = (df_filtrado['etapa'] == 'Etapa 3').astype(int)
    
    print(f"      Datos cargados. Total de registros a usar: {len(df_filtrado)}")

    # --- PASO 2: SEPARACION DE CARACTERISTICAS Y OBJETIVO ---
    print("[2/6] Separando matriz de caracteristicas (X) y variable objetivo (y)...")
    y = df_filtrado['objetivo_binario'].values
    X_crudo = df_filtrado.drop(columns=['etapa', 'objetivo_binario'])

    # --- PASO 3: DIVISION EN ENTRENAMIENTO Y PRUEBA (TRAIN/TEST SPLIT) ---
    print("[3/6] Dividiendo datos (80% Entrenamiento, 20% Prueba)...")
    X_train_crudo, X_test_crudo, y_train, y_test = train_test_split(
        X_crudo, y, test_size=0.20, stratify=y, random_state=42
    )

    # --- # PASO 4: PREPROCESAMIENTO (EL PIPELINE) ---
    print("[4/6] Iniciando tuberia de transformacion matematica...")
    pipeline = RiesgoCreditoPipeline()
    X_train_procesado = pipeline.fit_transform(X_train_crudo)
    X_test_procesado = pipeline.transform(X_test_crudo)

    # --- PASO 5: INSTANCIACION Y ENTRENAMIENTO DEL MODELO (LA ESTRATEGIA) ---
    print(f"[5/6] Iniciando algoritmo de aprendizaje: {args.modelo}...")
    
    # Seleccion de entrenamiento basada en el argumento de la terminal
    if args.modelo == 'red_neuronal':
        modelo = RedNeuronalEstrategia(epocas=50, batch_size=256, learning_rate=0.001)
        nombre_archivo_modelo = "modelo_red_neuronal.pth"
    else:
        modelo = RegresionLogisticaEstrategia()
        nombre_archivo_modelo = "modelo_regresion_logistica.joblib"
    
    modelo.entrenar(X_train_procesado, y_train)

    # --- PASO 6: PERSISTENCIA (GUARDADO EN DISCO) ---
    print("[6/6] Guardando artefactos para produccion...")
    ruta_directorio_modelos = "../models/"
    
    ruta_pipeline = os.path.join(ruta_directorio_modelos, "pipeline_preprocesamiento.joblib")
    ruta_modelo = os.path.join(ruta_directorio_modelos, nombre_archivo_modelo)
    
    pipeline.guardar(ruta_pipeline)
    modelo.guardar(ruta_modelo)
    
    print("=" * 60)
    print("PROCESO FINALIZADO EXITOSAMENTE. LOS MODELOS ESTAN LISTOS.")
    print("=" * 60)

if __name__ == "__main__":
    principal()