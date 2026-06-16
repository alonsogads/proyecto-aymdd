import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

class ConstructorDatasetHistorico:
    """
    Clase encargada de construir el dataset de muestra final a partir 
    de la base de datos cruda y sus catalogos dimensionales.
    Sigue el principio de Responsabilidad Unica: solo prepara datos historicos.
    """
    
    def __init__(self, ruta_hechos, ruta_catalogos, ruta_salida):
        self.ruta_hechos = ruta_hechos
        self.ruta_catalogos = ruta_catalogos
        self.ruta_salida = ruta_salida
        self.catalogos = {}
        
    def _cargar_catalogos(self):
        """
        Metodo privado que escanea el directorio de catalogos y los 
        carga dinamicamente en memoria.
        """
        print(" [Builder] Escaneando e ingiriendo catalogos dimensionales...")
        archivos_csv = glob.glob(os.path.join(self.ruta_catalogos, "*.csv"))
        
        if not archivos_csv:
            raise FileNotFoundError(f"No se encontraron catalogos en: {self.ruta_catalogos}")
            
        for archivo in archivos_csv:
            nombre_catalogo = os.path.basename(archivo).replace('.csv', '')
            self.catalogos[nombre_catalogo] = pd.read_csv(archivo, encoding='utf-8')
            
        print(f" [Builder] {len(self.catalogos)} catalogos cargados exitosamente.")

    def _integrar_dimensiones(self, df_hechos):
        """
        Metodo privado que realiza los cruces (Left Joins) para traducir
        las llaves numericas a texto interpretativo.
        """
        print(" [Builder] Iniciando proceso de desnormalizacion (Joins)...")
        df_integrado = df_hechos.copy()
        
        # Diccionario de mapeo de cruces estandar: (catalogo, llave_cruce, nueva_columna)
        cruces_estandar = [
            ('catalogo_sector_institucion', 'clave_institucion', 'institucion'),
            ('catalogo_entidad_federativa', 'clave_estado', 'estado'),
            ('catalogo_genero', 'clave_genero', 'genero'),
            ('catalogo_tipo_acreditado', 'clave_tipo_acreditado', 'tipo_acreditado'),
            ('catalogo_segmento_vivienda', 'clave_segmento_vivienda', 'segmento_vivienda'),
            ('catalogo_intervalo_edades', 'clave_intervalo_edades', 'intervalo_edades'),
            ('catalogo_sector_laboral', 'clave_sector_laboral', 'sector_laboral'),
            ('catalogo_destino_credito', 'clave_destino_credito', 'destino_credito'),
            ('catalogo_moneda', 'clave_moneda', 'moneda'),
            ('catalogo_etapas', 'clave_etapa', 'etapa')
        ]
        
        for nombre_cat, llave, nueva_col in cruces_estandar:
            df_integrado = pd.merge(df_integrado, self.catalogos[nombre_cat], on=llave, how='left')
            
        # Cruce con nombres de llave distintos (Ingreso Acreditado)
        df_integrado = pd.merge(
            df_integrado, 
            self.catalogos['catalogo_ingreso_acreditado'], 
            left_on='clave_intervalo_ingreso_acred', 
            right_on='clave_intervalo_ingreso_acreditado', 
            how='left'
        )
        # Renombramos la columna resultante para mantener consistencia
        df_integrado.rename(columns={'intervalo_ingreso_acreditado_y': 'intervalo_ingreso_acreditado'}, inplace=True, errors='ignore')
        
        return df_integrado

    def _muestreo_estratificado(self, df_integrado, tamano_muestra=1000000):
        """
        Extrae una muestra representativa garantizando la distribucion original de la variable objetivo.
        """
        print(f" [Builder] Ejecutando muestreo estratificado ({tamano_muestra} registros)...")
        
        # Si la base cruda es menor al tamano solicitado, usamos todo
        if len(df_integrado) <= tamano_muestra:
            return df_integrado
            
        df_muestra, _ = train_test_split(
            df_integrado, 
            train_size=tamano_muestra, 
            stratify=df_integrado['etapa'], 
            random_state=42
        )
        return df_muestra

    def _seleccionar_y_limpiar(self, df_muestra):
        """
        Filtra estrictamente las columnas requeridas por el pipeline de Machine Learning.
        """
        print(" [Builder] Seleccionando vector de caracteristicas final...")
        columnas_finales = [
            'intervalo_edades', 'intervalo_ingreso_acreditado', 'sector_laboral', 
            'estado', 'destino_credito', 'segmento_vivienda', 'moneda', 
            'saldo_insoluto_final_periodo', 'tasa_ponderada', 'etapa'
        ]
        
        # Validamos que todas las columnas existan antes de filtrar
        columnas_disponibles = [col for col in columnas_finales if col in df_muestra.columns]
        df_final = df_muestra[columnas_disponibles].copy()
        
        # Eliminamos nulos criticos en caso de que la fuente original tenga defectos atipicos
        df_final.dropna(subset=['etapa'], inplace=True)
        
        return df_final

    def construir_dataset(self):
        """
        Metodo publico (Orquestador). Ejecuta la cadena de valor completa 
        y guarda el resultado en disco.
        """
        print("=" * 60)
        print("CONSTRUCTOR DE DATASET HISTORICO")
        print("=" * 60)
        
        # 1. Cargar la base de hechos
        print(f" [Builder] Cargando tabla de hechos cruda desde: {self.ruta_hechos}")
        # Asumimos que la base historica puede ser un archivo gigante, especificamos low_memory=False
        df_hechos = pd.read_csv(self.ruta_hechos, low_memory=False)
        
        # 2. Cargar catalogos y cruzar
        self._cargar_catalogos()
        df_integrado = self._integrar_dimensiones(df_hechos)
        
        # 3. Reducir tamano y filtrar variables
        df_muestra = self._muestreo_estratificado(df_integrado)
        df_final = self._seleccionar_y_limpiar(df_muestra)
        
        # 4. Exportar a disco
        os.makedirs(os.path.dirname(self.ruta_salida), exist_ok=True)
        df_final.to_csv(self.ruta_salida, index=False)
        
        print(f"\n [EXITO] Dataset analitico construido y exportado en: {self.ruta_salida}")
        print(f"         Dimensiones finales: {df_final.shape[0]:,} filas x {df_final.shape[1]} columnas")
        print("=" * 60 + "\n")