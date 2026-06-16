import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Importamos nuestras herramientas construidas (Modelo)
from data.dataset_builder import ConstructorDatasetHistorico
from features.pipeline import RiesgoCreditoPipeline
from models.supervised import RedNeuronalEstrategia
from models.unsupervised import PamClusteringEstrategia

def principal():
    """
    Controlador principal. Ejecuta la construccion de datos y entrena AMBOS
    modelos (Supervisado y No Supervisado) en una sola ejecucion.
    """
    print("=" * 60)
    print("CONSTRUCCION DE DATOS Y ENTRENAMIENTO DE MODELOS - RIESGO CREDITICIO")
    print("=" * 60)

    # --- PASO 1: CARGA O CONSTRUCCION DE DATOS ---
    ruta_datos_crudos = "../data/Base_Historica/Base_Historica/Base_Historica_Portafolio_Total.csv" 
    ruta_catalogos = "../data/Base_Historica/Catalogos/"
    ruta_datos_limpios = "../data/Datos_Procesados/dataset_hipotecario_limpio.csv"
    
    print("[1/7] Verificando existencia de dataset procesado...")
    if not os.path.exists(ruta_datos_limpios):
        print("      Dataset limpio no encontrado. Iniciando construccion automatica...")
        constructor = ConstructorDatasetHistorico(
            ruta_hechos=ruta_datos_crudos,
            ruta_catalogos=ruta_catalogos,
            ruta_salida=ruta_datos_limpios
        )
        constructor.construir_dataset()
    else:
        print(f"      Dataset limpio detectado en: {ruta_datos_limpios}")
        
    df = pd.read_csv(ruta_datos_limpios)
    df_filtrado = df[df['etapa'].isin(['Etapa 1', 'Etapa 3'])].copy()
    df_filtrado['objetivo_binario'] = (df_filtrado['etapa'] == 'Etapa 3').astype(int)

    # --- PASO 2: SEPARACION DE CARACTERISTICAS Y OBJETIVO ---
    print("[2/7] Separando variables para modelo predictivo...")
    y = df_filtrado['objetivo_binario'].values
    X_crudo = df_filtrado.drop(columns=['etapa', 'objetivo_binario'])

    # --- PASO 3: DIVISION TRAIN/VAL/TEST ---
    print("[3/7] Dividiendo datos (64% Train, 16% Val, 20% Test)...")
    X_train_temp, X_test_crudo, y_train_temp, y_test = train_test_split(
        X_crudo, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train_crudo, X_val_crudo, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=0.20, stratify=y_train_temp, random_state=42
    )

    # --- PASO 4: PREPROCESAMIENTO (PIPELINE) ---
    print("[4/7] Entrenando tuberia de preprocesamiento estadistico...")
    pipeline = RiesgoCreditoPipeline()
    X_train_procesado = pipeline.fit_transform(X_train_crudo)
    X_val_procesado = pipeline.transform(X_val_crudo)

    # --- PASO 5: ENTRENAMIENTO RED NEURONAL (SUPERVISADO) ---
    print("[5/7] Entrenando modelo supervisado (Red Neuronal)...")
    modelo_nn = RedNeuronalEstrategia(epocas=30, batch_size=256, learning_rate=0.001)
    modelo_nn.entrenar(X_train_procesado, y_train, X_val_procesado, y_val)

    # --- PASO 6: ENTRENAMIENTO PAM CLUSTERING (NO SUPERVISADO) ---
    print("\n[6/7] Descubriendo perfiles desde el modelo no supervisado (PAM Clustering)...")
    # Filtramos estrictamente la cartera vencida para buscar los perfiles reales
    df_mora = df_filtrado[df_filtrado['etapa'] == 'Etapa 3'].copy()
    
    # Muestreo a 10,000 para viabilidad matematica de Gower (O(N^2))
    if len(df_mora) > 10000:
        df_mora_muestra = df_mora.sample(n=10000, random_state=42)
    else:
        df_mora_muestra = df_mora
        
    columnas_clustering = [
        'intervalo_edades', 'intervalo_ingreso_acreditado', 'sector_laboral', 
        'estado', 'destino_credito', 'segmento_vivienda', 'moneda', 
        'saldo_insoluto_final_periodo', 'tasa_ponderada'
    ]
    X_clustering = df_mora_muestra[columnas_clustering]
    
    modelo_pam = PamClusteringEstrategia(n_clusters=2)
    modelo_pam.entrenar(X_clustering)

    # --- PASO 7: PERSISTENCIA EN DISCO ---
    print("\n[7/7] Guardando artefactos en disco...")
    ruta_dir = "../models/"
    os.makedirs(ruta_dir, exist_ok=True)
    
    pipeline.guardar(os.path.join(ruta_dir, "pipeline_preprocesamiento.joblib"))
    modelo_nn.guardar(os.path.join(ruta_dir, "modelo_red_neuronal.pth"))
    modelo_pam.guardar(os.path.join(ruta_dir, "modelo_pam_clustering.joblib"))
    
    print("=" * 60)
    print("PROCESO FINALIZADO EXITOSAMENTE. LISTO PARA EJECUTAR demo.py")
    print("=" * 60)

if __name__ == "__main__":
    principal()