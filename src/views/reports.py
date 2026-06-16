from sklearn.metrics import classification_report, confusion_matrix

def mostrar_encabezado(titulo):
    """
    Imprime un encabezado estandarizado para mantener una interfaz
    de usuario (UI) limpia en la terminal.
    """
    print("\n" + "=" * 60)
    print(f"{titulo.upper().center(60)}")
    print("=" * 60)

def mostrar_resultados_evaluacion(y_real, y_prediccion):
    """
    Toma las etiquetas reales y las predicciones del modelo y las formatea
    en un reporte estadistico legible.
    """
    mostrar_encabezado("Reporte de Clasificacion")
    # Generamos el reporte de Scikit-Learn pero lo imprimimos estructuradamente
    reporte = classification_report(
        y_real, 
        y_prediccion, 
        target_names=['Sano (Etapa 1)', 'Mora (Etapa 3)']
    )
    print(reporte)
    
    mostrar_encabezado("Matriz de Confusion (Impacto de Negocio)")
    matriz = confusion_matrix(y_real, y_prediccion)
    
    print("Desglose de predicciones:")
    print(f" -> Verdaderos Negativos (Sanos correctos): {matriz[0][0]}")
    print(f" -> Falsos Positivos (Sanos clasificados como Mora - Oportunidad perdida): {matriz[0][1]}")
    print(f" -> Falsos Negativos (Mora clasificada como Sano - Riesgo financiero): {matriz[1][0]}")
    print(f" -> Verdaderos Positivos (Mora correcta - Riesgo mitigado): {matriz[1][1]}\n")

def mostrar_prediccion_cliente(es_riesgo):
    """
    Muestra de forma destacada el resultado final de la evaluacion
    de un cliente individual en el simulador.
    """
    mostrar_encabezado("RESULTADO DE LA EVALUACION DE RIESGO")
    
    if es_riesgo == 1:
        print(" [ ! ] ALERTA: EL CLIENTE PRESENTA ALTO RIESGO DE IMPAGO (ETAPA 3)")
        print(" Recomendacion: Rechazar solicitud o turnar a comite para garantias adicionales.")
    else:
        print(" [ v ] APROBADO: EL CLIENTE TIENE UN PERFIL FINANCIERO SANO (ETAPA 1)")
        print(" Recomendacion: Continuar con el proceso de originacion estandar.")
        
    print("=" * 60 + "\n")

def mostrar_perfil_cluster(cluster_id):
    """
    Muestra la interpretacion de negocio del modelo no supervisado
    cuando a un cliente en mora se le asigna un grupo.
    """
    mostrar_encabezado("ASIGNACION DE PERFIL DE MOROSIDAD")
    
    if cluster_id == 0:
        print(" -> PERFIL 0: Morosidad por Expansion")
        print("    (Caracteristicas: Jovenes, Vivienda Nueva, Estado de Mexico)")
    elif cluster_id == 1:
        print(" -> PERFIL 1: Morosidad por Vulnerabilidad")
        print("    (Caracteristicas: Maduros, Vivienda Usada, CDMX)")
    else:
        print(f" -> ASIGNADO AL CLUSTER: {cluster_id}")
        
    print("=" * 60 + "\n")

def mostrar_datos_ingresados(diccionario_datos):
    """
    Muestra en pantalla los datos crudos capturados para confirmacion visual.
    """
    mostrar_encabezado("DATOS CAPTURADOS DEL CLIENTE")
    for llave, valor in diccionario_datos.items():
        # Formateamos la salida para que se vea como una tabla de dos columnas
        print(f" * {llave.replace('_', ' ').capitalize():<30}: {valor}")
    print("-" * 60)