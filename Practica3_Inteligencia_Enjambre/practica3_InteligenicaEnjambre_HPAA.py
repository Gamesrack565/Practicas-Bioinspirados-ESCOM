#Algoritmos bionspirados
#Profesora: Abril Valeria Uriarte Arcia

#Practica 3: Inteligencia de Enjambre

#By: @Angel Abraham HIguera Pineda
#Grupo: 5BM1

#Librerias:
#Generacion de numeros aleatorios
import random
#Para acceder a parametros y funciones especificos del sistema, como 'float_info'
import sys
#Para operaciones numericas y manejo de arreglos, con el alias 'np'
import numpy as np
#Importa la biblioteca 'multiprocessing' para procesamiento paralelo
#--- REALMENTE NO ES NECESARIO
import multiprocessing as mp


#Funciones:

#Define la funcion objetivo que se va a minimizar
def funcion_objetiva(coordenadas_part):
    x = coordenadas_part[0]
    y = coordenadas_part[1]
    return x**2 + y**2 + (25 * (np.sin(x) + np.sin(y)))

#Funcion para imprimir de forma detallada los datos de una iteracion especifica
def mostrar_detalles_iteracion(data, total_iters):
    print(f"\n-- ITERACION {data['Itearcion:']} / {total_iters} --")
    print("\n--- Posicion de las partículas (x, y) ---")
    print(data['Posiciones:'])
    print("\n--- Velocidad de las partículas (vx, vy) ---")
    print(data['Velocidades:'])
    print("\n--- Mejores Posiciones Personales (pbest) ---")
    print(data['Pbest:'])
    print("\n--- Mejor Posición Global (gbest) ---")
    print(f"Posición (x, y): {data['Gbest_posi:']}")
    print(f"Valor (mínimo): {data['Gbest_val:']:.4f}")

#Funcion principal
#Es necesario mantenerlo en una funcion, para evitar que se genere un error con el multiprocessing
def programa_Pso():
    #Parametros:
    Num_particulas = 20
    Num_iteraciones = 50
    #Define el numero de dimensiones del problema (en este caso, 'x' y 'y')
    Num_variables = 2

    #Intervalos:
    min_intervalo = -5.0
    max_intervalo = 5.0

    #Inercia:
    a = 0.8
    #Aprendizaje local
    B1 = 0.7
    #Aprendizaje global
    B2 = 1.2


    #Inicializa las posiciones de las particulas con valores aleatorios dentro del intervalo
    particula_posi = np.random.uniform(min_intervalo,max_intervalo,(Num_particulas,Num_variables))
    #Inicializa las velocidades de todas las particulas a cero
    particula_veloci = np.zeros((Num_particulas, Num_variables))
    #Inicializa las mejores posiciones (pbest) como una copia de las posiciones iniciales
    particula_pbest_posi = np.copy(particula_posi)
    #Inicializa los valores de pbest al valor maximo posible (infinito) para un problema de minimizacion
    particula_pbest_val = np.full(Num_particulas, sys.float_info.max)

    #Inicia un bucle para recorrer cada particula
    for i in range(Num_particulas):
        #Calcula y asigna el valor (fitness) inicial para la pbest de cada particula
        particula_pbest_val[i] = funcion_objetiva(particula_posi[i])

    #Encuentra el indice de la particula con el mejor valor pbest (el minimo) de todos
    gbest_index = np.argmin(particula_pbest_val)
    #Guarda la posicion de esa mejor particula como la mejor posicion global (gbest)
    gbest_posi = np.copy(particula_pbest_posi[gbest_index])
    #Guarda el valor (fitness) de esa mejor particula como el mejor valor global
    gbest_val = np.copy(particula_pbest_val)[gbest_index]

    gbest_veloci = np.copy(particula_veloci[gbest_index])

    print("Inicio del programa.....")
    print("FUNCION A MINIMIZAR: x^2 + y^2 + (25 * (sen(x) + sen(y)))")
    print(f"Inicando PSO Global con {Num_particulas} particulas y {Num_iteraciones} iteraciones....")
    print(f"Parametros: \n* A= {a}, \n* B1= {B1}, \n* B2= {B2}")

    print("\n--------------")
    print("ADVERTENCIA: Los Pbest se deben ver en mas informacion")
    print("--------------\n")

    #Crea una lista vacia para guardar los datos de cada iteracion
    historial = []

    #Inicia un 'pool' de procesos de multiprocessing para calculos paralelos
    with mp.Pool() as pool:
        #Inicia el bucle principal de las iteraciones del algoritmo
        for l in range(Num_iteraciones):
            #Inicia un bucle interno para actualizar cada particula
            for i in range(Num_particulas):
                #Variables r1 y r2, da numeros entre 0 y 1, para la funcion principal
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)

                #Calcula el componente de inercia de la nueva velocidad
                parte1_funcion = a * particula_veloci[i]
                #Calcula el componente cognitivo (local) de la velocidad
                parte2_funcion = B1 * r1 * (particula_pbest_posi[i] - particula_posi[i])
                #Calcula el componente social (global) de la velocidad
                parte3_funcion = B2 * r2 * (gbest_posi - particula_posi[i])

                #particula_veloci[l] = (a * particula_veloci) + (B1 * r1 * (particula_posi[i] - particula_posi[i])) + (B2 * r2 * (gbest_posi[i] - particula_posi[i]))

                #Suma los tres componentes para obtener la nueva velocidad
                new_veloci = parte1_funcion + parte2_funcion + parte3_funcion
                #Actualiza la velocidad de la particula
                particula_veloci[i] = new_veloci

                #Calcula la nueva posicion de la particula sumando la nueva velocidad
                new_posi = particula_posi[i] + new_veloci
                #Asegura que la nueva posicion se mantenga dentro de los limites del intervalo
                new_posi = np.clip(new_posi, min_intervalo, max_intervalo)
                #Actualiza la posicion de la particula
                particula_posi[i] = new_posi

            #Usa el pool de multiprocessing para evaluar la funcion objetivo en todas las particulas (en paralelo)
            new_valores = pool.map(funcion_objetiva, particula_posi)

            #Inicia un bucle para revisar los nuevos valores de fitness de cada particula
            for z in range(Num_particulas):
                #Comprueba si el nuevo valor de la particula es mejor que su pbest actual
                if new_valores[z] < particula_pbest_val[z]:
                    #Si es mejor, actualiza su valor pbest
                    particula_pbest_val[z] = new_valores[z]
                    #Y tambien actualiza su posicion pbest
                    particula_pbest_posi[z] = particula_posi[z]

            #Despues de actualizar todos los pbest, encuentra el indice del mejor pbest
            mejor_index = np.argmin(particula_pbest_val)
            #Comprueba si este valor es mejor que el gbest actual
            if particula_pbest_val[mejor_index] < gbest_val:
                #Si es mejor, actualiza el valor gbest (valor global)
                gbest_val = particula_pbest_val[mejor_index]
                #Y actualiza la posicion gbest
                gbest_posi = particula_pbest_posi[mejor_index]
                gbest_veloci = np.copy(particula_veloci[mejor_index])


            #Anade un diccionario con los datos de la iteracion actual al historial
            historial.append({
                "Itearcion:": l + 1,
                "Posiciones:": np.copy(particula_posi),
                "Velocidades:": np.copy(particula_veloci),
                "Pbest:": np.copy(particula_pbest_posi),
                "Gbest_posi:": np.copy(gbest_posi),
                "Gbest_val:": gbest_val
            })

            #Imprime el mejor de cada iteracion
            #print(r1)
            print(f"\n -- ITERACION {l + 1} / {Num_iteraciones} --")
            print(f"* Mejor posicion (gbest_posicion): {gbest_posi}")
            print(f"* Valor minimo (gbest_valor): {gbest_val:}")
            print(f"* Velocidad: {gbest_veloci}")



    #Imprime el mejor resultado despues de las 50 iteraciones
    print("\n\n--- CARGANDO RESULTADO..... ---\n")
    print(f"Trabajo completado en {Num_iteraciones} iteraciones.")
    print(f"Mejor posicion encontrada: {gbest_posi}")
    print(f"Valor minimo encontrado: {gbest_val:.4f}")
    print(f"Velocidad encontrada: {gbest_veloci}")

    #Seccion de mas detalles
    print("\n\n- MAYOR INFORMACION -\n")
    opcion = input("Quieres ver los detalles de las iteraciones? (S/N): ").strip().lower()
    #Comprueba si la respuesta del usuario fue 's' (si)
    if opcion == 's':
        sub_opcion = input("Desea ver (T) todas o (U) una iteración específica?: ").strip().lower()

        #Comprueba si la respuesta fue t (todas)
        if sub_opcion == 't':
            #Mostrar todas las iteraciones
            #Inicia un bucle que recorre cada elemento guardado en el historial
            for data in historial:
                #Llama a la funcion de impresion detallada para cada iteracion
                mostrar_detalles_iteracion(data, Num_iteraciones)
        #Si la respuesta fue u (una)
        elif sub_opcion == 'u':
            try:
                #Pide al usuario que ingrese el numero de iteracion deseado
                num = int(input(f"Ingrese el número de iteración (1-{Num_iteraciones}): "))
                #Comprueba si el numero ingresado esta dentro del rango valido de iteraciones
                if 1 <= num <= Num_iteraciones:
                    #Muestra los detalles de la iteracion solicitada (ajustando el indice base 1 a base 0)
                    mostrar_detalles_iteracion(historial[num - 1], Num_iteraciones)
                #Si el numero esta fuera de rango
                else:
                    #Imprime un mensaje de error
                    print("---- NUMERO INDICADO FUERA DEL RANGO DE ITERACIONES ----")
            #Manejo de rrores
            except ValueError:
                #Imprime un mensaje de error
                print("---- ENTRADA NO VALIDA ----")
        #SManejo de errores
        else:
            #Imprime un mensaje de error y termina
            print("Opción no valida. Finalizando......")

#Necesario para la ejecucion del codigo
if __name__ == "__main__":
    #Anade soporte para multiprocessing en entornos congelados
    #Ya no es necesario
    mp.freeze_support()
    print("---------------------------------------")
    print("Practica 3: Inteligencia de Enjambre")
    print("By: Angel Abraham Higuera Pineda")
    print("Grupo: 5BM1")
    print("---------------------------------------\n\n")

    programa_Pso()