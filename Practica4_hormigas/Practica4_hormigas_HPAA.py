#Algoritmos bionspirados
#Profesora: Abril Valeria Uriarte Arcia

#Practica 4: Colonia de hormigas

#By: @Angel Abraham Higuera Pineda
#Grupo: 5BM1

#libreria:
import numpy as np


#Prorgrama:

#Inicializacion de la matriz
distancias = np.array([
    [0, 6, 9, 17, 13, 21],
    [6, 0, 19, 21, 12, 18],
    [9, 19, 0, 20, 23, 11],
    [17, 21, 20, 0, 15, 10],
    [13, 12, 23, 15, 0, 21],
    [21, 18, 11, 10, 21, 0]
], dtype=float)

#Parametros:
num_ciudades = 6
num_hormigas = 6          #Una hormiga por nodo
iteraciones = 50
a = 1.5                   #Para la intesidad de la feromona (τ(i,j)^α)
b = 0.8                   #Para la visibilidad (η(i,j)^b)
p = 0.2                   #Para la operacion 1 - p

#FEROMONAS
#Inicializacion de la matriz
#Nunca debe ser 0
feromonas = np.ones((num_ciudades, num_ciudades)) * 0.1

#Diagonal de las feromonas
np.fill_diagonal(feromonas, 0)

#MATRIZ DE VISIBILIDAD
with np.errstate(divide='ignore'):
    #Inversa de la matriz
    #n(i,j)
    visibilidad = 1.0 / distancias
    visibilidad[np.isinf(visibilidad)] = 0

#Matriz de las potencias de la visibilidad
#Potencia n(i,j)^b
#Se llama al resultado en el cliclo, no es necesario calcularlo en todos los ciclos
visibilidad_potencia = np.power(visibilidad, b)

#BUCLE PRINCIPAL

mejor_ruta_global = []
mejor_distancia_global = float('inf')


print("---------------------------------------")
print("Practica 4: Enjambre de hormigas")
print("By: Angel Abraham Higuera Pineda")
print("Grupo: 5BM1")
print("---------------------------------------\n\n")

print(f"{'Iter':<5} | {'Mejor Ruta de la Iteración':<30} | {'Distancia'}")
print("----------------------------------------------------------")

for it in range(iteraciones):
    #Guardado de rutas y distancias de la iteración
    rutas_iteracion = np.zeros((num_hormigas, num_ciudades + 1), dtype=int)
    distancias_iteracion = np.zeros(num_hormigas)

    #Matriz de feromonas copiada y evaluada a 0
    #lista para actualizar las nuevas feromonas
    #MAtriz temporal
    feromonas_temporal = np.zeros_like(feromonas)

    #CONSTRUCCION DE RUTAS
    for k in range(num_hormigas):
        #Cada hormiga inicia en un nodo distinto
        actual = k
        rutas_iteracion[k, 0] = actual
        #Indica que no se ha visitado ninguna ciudad
        visitados = np.zeros(num_ciudades, dtype=bool)
        #Donde esat se marcara como visitado (el nodo inicial)
        visitados[actual] = True
        #Como inicia, no hay ninguna distancia recorrida
        distancia_total = 0

        #RECORRIDO
        for paso in range(1, num_ciudades):
            #CALCULO DE PROBABILIDADES
            #P(i,j) = ( τ(i,j)^α * η(i,j)^b ) / sum( τ(i,l)^α * η(i,l)^b )

            #Multiplicaciones
            #τ(i,j)^α
            parte1 = np.power(feromonas[actual], a)

            #η(i,j)^β ya pre-calculado
            parte2 = visibilidad_potencia[actual]

            #Numerador completo de la fórmula
            probs_individuales = parte1 * parte2

            #Proteccion de nodos visitados
            #Los marca en 0, para que ya no los visite
            probs_individuales[visitados] = 0

            #Denominador  (Sumatoria completa)
            suma_probs = np.sum(probs_individuales)

            #RULETA
            if suma_probs > 0:
                #Probabilidades
                probs_norm = probs_individuales / suma_probs

                #Creamos la ruleta
                suma_acumulada = np.cumsum(probs_norm)
                #Random
                r = np.random.rand()
                #Busca el numero si es mayor o igual
                siguiente = np.searchsorted(suma_acumulada, r)
            else:
                #Caso de que tenga un error
                #Obtiene un camnio al azar, para que no se detenga la hormiga
                no_visitados = np.where(~visitados)[0]
                #EN CASO DE QUE UN SUPER ERRROR
                siguiente = np.random.choice(no_visitados)

            #Movimiento de la hormiga
            rutas_iteracion[k, paso] = siguiente
            visitados[siguiente] = True
            #Va acumulando el valor de la distancia para al final ver si es una buena distanica
            distancia_total += distancias[actual, siguiente]
            actual = siguiente


        #Regreso al inicio
        #Recuperamos la ciudad donde esta hormiga empezó su viaje
        nodo_inicial = rutas_iteracion[k, 0]
        #Agregamos el nodo inicial al final de la lista para cerrar visualmente la ruta
        #Ejemplo: [3, 1, 4, 2, 5, 6] se convierte en [3, 1, 4, 2, 5, 6, 3]
        rutas_iteracion[k, -1] = nodo_inicial
        #Sumamos la distancia del último tramo: desde la ciudad actual de regreso al inicio
        distancia_total += distancias[actual, nodo_inicial]
        #Guardamos la calificación final de esta hormiga
        distancias_iteracion[k] = distancia_total

        #Invera del costo total para las feromonas
        aporte = 1 / distancia_total
        #Obtiene el camino
        ruta = rutas_iteracion[k]

        #Realizamos la Σ_k Δτ_k(i,j)
        for i in range(num_ciudades):
            u, v = ruta[i], ruta[i + 1]
            # i varias hormigas usaron la arista (u,v), sus aportes se SUMAN en esta casilla.
            feromonas_temporal[u, v] += aporte
            feromonas_temporal[v, u] += aporte

    #ACTUALIZACION DE LAS FEROMONAS
    # τ(i,j) = (1 - ρ) * τ(i,j) + Σ_k Δτ_k(i,j)

    #Actualizar feromonas
    feromonas = (1 - p) * feromonas + feromonas_temporal

    #Mantener diagonal en cero
    np.fill_diagonal(feromonas, 0)

    #RESULTADOS
    #1. Encontrar la mejor hormiga de ESTA iteración (Mejor Local)
    #np.argmin nos dice el ÍNDICE de la hormiga que recorrió menos distancia
    min_idx = np.argmin(distancias_iteracion)

    #2. Actualizar el Mejor Global
    #Si la mejor hormiga de hoy superó al mejor récord que teníamos guardado...
    if distancias_iteracion[min_idx] < mejor_distancia_global:
        mejor_distancia_global = distancias_iteracion[min_idx]
        #Usamos .copy() para guardar una COPIA de la ruta.
        #Si no lo usáramos, la ruta se borraría en la siguiente vuelta del bucle.
        mejor_ruta_global = rutas_iteracion[min_idx].copy()

    #3. Mostramos el resultado
    ruta_resultados = (rutas_iteracion[min_idx] + 1).tolist()
    print(f"{it + 1:<5} | {str(ruta_resultados):<30} | {distancias_iteracion[min_idx]:.4f}")

print("----------------------------------------------------------")
print("RESULTADO FINAL OPTIMO:")
print(f"Ruta: {(mejor_ruta_global + 1).tolist()}")
print(f"Distancia Mínima: {mejor_distancia_global}")