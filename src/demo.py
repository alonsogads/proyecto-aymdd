import os
import pandas as pd
import warnings

# Ocultamos advertencias de scikit-learn sobre nombres de caracteristicas para una interfaz limpia
warnings.filterwarnings("ignore")

from features.pipeline import RiesgoCreditoPipeline
from models.supervised import RedNeuronalEstrategia
from models.unsupervised import PamClusteringEstrategia
from views import reports

def capturar_seleccion(mensaje, opciones):
    """
    Muestra un menu numerado y fuerza al usuario a elegir una opcion valida.
    """
    print(f"\n{mensaje}")
    for i, opcion in enumerate(opciones):
        print(f"  {i + 1}. {opcion}")
        
    while True:
        try:
            seleccion = int(input(" -> Ingresa el numero de tu eleccion: "))
            if 1 <= seleccion <= len(opciones):
                return opciones[seleccion - 1]
            else:
                print(" [!] Opcion fuera de rango. Intenta de nuevo.")
        except ValueError:
            print(" [!] Entrada invalida. Por favor ingresa un numero.")

def capturar_datos_cliente():
    """
    Simula el formulario de captura que llenaria un ejecutivo del banco.
    Mapea estrictamente hacia los catalogos de datos limpios.
    """
    reports.mostrar_encabezado("DEMOSTRACION PARA MODELOS DE RIESGO DE CREDITO")
    print("Por favor, complete la siguiente informacion del solicitante:\n")

    # 1. Variables Numericas
    while True:
        try:
            saldo = float(input("1. Saldo Insoluto / Monto del Credito (MXN): "))
            tasa = float(input("2. Tasa de Interes Ponderada (ej. 10.5): "))
            break
        except ValueError:
            print(" [!] Por favor ingresa valores numericos validos.")

    # 2. Variables Categoricas / Ordinales (Menus)
    edad = capturar_seleccion(
        "3. Seleccione el Intervalo de Edad:",
        ['Menor a 26', '26-35', '36-45', '46-55', '56-65', 'Mas de 65']
    )
    
    ingreso = capturar_seleccion(
        "4. Seleccione el Intervalo de Ingreso Mensual:",
        ['Menos de 10,000 MXN', '(10,001 - 20,000)', '(20,001 - 40,000)', 
         '(40,001 - 80,000)', '(80,001 - 150,000)', 'Mas de 150,000 MXN']
    )
    
    sector = capturar_seleccion(
        "5. Seleccione el Sector Laboral:",
        ['Sector privado', 'Sector publico estatal', 'No asalariado', 
         'Sin clasificar']
    )
    
    estado = capturar_seleccion(
        "6. Seleccione el Estado de ubicacion del Inmueble:",
        [
            'AGUASCALIENTES', 'BAJA CALIFORNIA', 'BAJA CALIFORNIA SUR', 'CAMPECHE',
            'CHIAPAS', 'CHIHUAHUA', 'CIUDAD DE MEXICO', 'COAHUILA DE ZARAGOZA',
            'COLIMA', 'DURANGO', 'GUANAJUATO', 'GUERRERO', 'HIDALGO', 'JALISCO',
            'MEXICO', 'MICHOACAN DE OCAMPO', 'MORELOS', 'NAYARIT', 'NUEVO LEON',
            'OAXACA', 'PUEBLA', 'QUERETARO', 'QUINTANA ROO', 'SAN LUIS POTOSI',
            'SINALOA', 'SONORA', 'TABASCO', 'TAMAULIPAS', 'TLAXCALA',
            'VERACRUZ DE IGNACIO DE LA LLAVE', 'YUCATAN', 'ZACATECAS'
        ]
    )
    
    destino = capturar_seleccion(
        "7. Seleccione el Destino del Credito:",
        ['Adquisicion de Vivienda Nueva', 'Adquisicion de Vivienda Usada', 'Adquisición de Terreno para Vivienda',
         'Adquisición de Terreno y Construcción Simultánea', 'Construcción de Vivienda Propia',
         'Mejoras, Ampliaciones y/o Remodelaciones', 'Crédito Para Líquidez', 'Créditos a exempleados de la Entidad',
         'Pago de Pasivos Hipotecarios', 'Autoproducción de vivienda']
    )
    
    segmento = capturar_seleccion(
        "8. Seleccione el Segmento de la Vivienda:",
        ['Media o Residencial', 'Interes Social', 'Cartera adquirida al INFONAVIT o el FOVISSSTE', 
         'Régimen especial de amortización', 'Remodelación o mejoramiento con garantía de la subcuenta de vivienda', 
         'Remodelación o mejoramiento con gtía otorg Banca Desarrollo o fideic públicos', 'Otro']
    )
    
    moneda = capturar_seleccion(
        "9. Seleccione la Moneda de Originación de Credito:",
        ['Moneda Nacional (Pesos)', 'VSMG (Veces Salario Minimo General)', 'UDIS', 'Dólares de E.E.U.U.A.', 
         'UMA (Unidad de medida y actualización)', '	Sin clasificar']
    )

    # Construimos el diccionario con las llaves exactas que espera el DataFrame
    datos_cliente = {
        'intervalo_edades': edad,
        'intervalo_ingreso_acreditado': ingreso,
        'sector_laboral': sector,
        'estado': estado,
        'destino_credito': destino,
        'segmento_vivienda': segmento,
        'moneda': moneda,
        'saldo_insoluto_final_periodo': saldo,
        'tasa_ponderada': tasa
    }
    
    return datos_cliente

def principal():
    """
    Orquestador. Carga los modelos, procesa el input del usuario
    y emite un dictamen en cascada (Supervisado -> No Supervisado).
    """
    # --- FASE 1: CAPTURA DE DATOS ---
    diccionario_datos = capturar_datos_cliente()
    reports.mostrar_datos_ingresados(diccionario_datos)
    
    # Convertimos a DataFrame (1 sola fila) para compatibilidad con el sistema
    df_cliente = pd.DataFrame([diccionario_datos])
    
    # --- FASE 2: CARGA DE ARTEFACTOS Y TRANSFORMACION ---
    print("\n[Sistema] Cargando datos de los modelos...\n")
    
    ruta_modelos = "../models/"
    
    try:
        # Carga y aplicacion del Pipeline
        pipeline = RiesgoCreditoPipeline.cargar(os.path.join(ruta_modelos, "pipeline_preprocesamiento.joblib"))
        X_procesado = pipeline.transform(df_cliente)
        
        # Carga de Modelos
        # Nota: Le pasamos input_dim calculandolo al vuelo desde nuestra matriz procesada
        modelo_supervisado = RedNeuronalEstrategia.cargar(
            os.path.join(ruta_modelos, "modelo_red_neuronal.pth"), 
            input_dim=X_procesado.shape[1]
        )
        
        modelo_no_supervisado = PamClusteringEstrategia.cargar(
            os.path.join(ruta_modelos, "modelo_pam_clustering.joblib")
        )
        
    except Exception as e:
        print(f" [!] Error critico al cargar los modelos: {str(e)}")
        return

    # --- FASE 3: INFERENCIA Y TOMA DE DECISIONES ---
    
    # Paso A: El filtro predictivo (Red Neuronal)
    prediccion_riesgo = modelo_supervisado.predecir(X_procesado)[0]
    
    # Mostramos el dictamen inicial
    reports.mostrar_prediccion_cliente(prediccion_riesgo)
    
    # Paso B: El diagnostico descriptivo (K-Medoids)
    # SOLO se ejecuta si la red neuronal predijo riesgo (1)
    if prediccion_riesgo == 1:
        print("\n[Sistema] Comenzando analisis con perfiles de riesgo...\n")
        
        # OJO: K-Medoids y Gower esperan el dataframe CRUDO, no el procesado
        perfil_asignado = modelo_no_supervisado.predecir(df_cliente)[0]
        
        # Mostramos la radiografia del perfil
        reports.mostrar_perfil_cluster(perfil_asignado)

if __name__ == "__main__":
    principal()