# Angel Abraham Higuera Pineda
# Practica: Abejas
# Profesora: Abril Valeria Uriarte Arcia


import random
import copy
from typing import List

#Configuraciones solicitadas
#Limite de mochila
CAPACIDAD_MAXIMA = 30

#Diccionario de los productos
productos_data = [
    {"nombre": "Decoy Detonators", "peso": 4, "precio": 10, "min": 0, "max": 10},
    {"nombre": "Fever Fudge", "peso": 2, "precio": 3, "min": 0, "max": 10},
    {"nombre": "Love Potion", "peso": 2, "precio": 8, "min": 3, "max": 10},
    {"nombre": "Puking Pastilles", "peso": 1.5, "precio": 2, "min": 0, "max": 10},
    {"nombre": "Extendable Ears", "peso": 5, "precio": 12, "min": 0, "max": 10},
    {"nombre": "Nosebleed Nougat", "peso": 1, "precio": 2, "min": 0, "max": 10},
    {"nombre": "Skiving Snackbox", "peso": 5, "precio": 6, "min": 2, "max": 10}
]

#Parametros indicados en la practica
TAMANO_ENJAMBRE = 40      #Total de fuentes de alimento
NUM_OBRERAS = 20          #Mitad del enjambre
NUM_OBSERVADORAS = 20     #La otra mitad del enjambre
LIMITE = 5                #Limite
ITERACIONES = 50


class Solucion:
    def __init__(self):
        self.cantidades = []    # Cromosoma: cantidad de cada producto
        self.fitness = 0.0      # Aptitud: calidad de la solucion
        self.peso_total = 0.0   # Peso acumulado
        self.valor_total = 0.0  # Beneficio economico total
        self.intentos = 0       # Contador de estancamiento


    # FUNCION DE INICIALIZACION DE ABEJAS
    def inicializar(self):
        self.cantidades = []
        for p in productos_data:
            #Obtenemos los valores y los pasamos a variables
            #Para hacer uso de la formula
            l_j = p["min"]  # Limite inferior
            u_j = p["max"]  # Limite superior
            r1 = random.random()  # r1 es un valor aleatorio entre [0, 1)

            #Aplicamos la formula: x_ij = l_j + r1 * (u_j - l_j)
            valor_calculado = l_j + r1 * (u_j - l_j)

            #Adaptacion a Enteros (Discretizacion)
            #Como no podemos vender 3.4 pociones, redondeamos
            cant = int(round(valor_calculado))

            self.cantidades.append(cant)

        #Pasamos a evaluar
        self.evaluar()

    def evaluar(self):
        self.peso_total = 0
        self.valor_total = 0
        #Recorremos el cromosoma calculando totales
        for i, p in enumerate(productos_data):
            self.peso_total += self.cantidades[i] * p["peso"]
            self.valor_total += self.cantidades[i] * p["precio"]

        #FUNCION DE APTITUD (Fitness)
        if self.peso_total <= CAPACIDAD_MAXIMA:
            #Si la solucion es valida (no excede peso), el fitness es su valor monetario.
            #Como los precios son positivos, esto sera > 1.
            self.fitness = self.valor_total
        else:
            #Si excede el peso, aplicamos una penalizacion fuerte.
            #Fitness sera un valor muy bajo (entre 0 y 1).
            sobrepeso = self.peso_total - CAPACIDAD_MAXIMA
            self.fitness = 1.0 / (1.0 + sobrepeso)

#RULETA en version funcion
def seleccion_ruleta(aptitudes: List[float]) -> int:
    #Usamos este metodo, ya que el metodo corto usado en la practica anterior
    #No es tan facil de entender como trabaja
    total = sum(aptitudes)
    #Si todas las aptitudes son 0 (caso raro), devuelve uno al azar
    if total <= 0: return random.randint(0, len(aptitudes) - 1)

    r = random.uniform(0, total)
    acumulado = 0.0
    for i, f in enumerate(aptitudes):
        acumulado += f
        if acumulado >= r: return i
    return len(aptitudes) - 1


def generar_vecino(poblacion, indice_actual):
    actual = poblacion[indice_actual]

    #1. Seleccionar vecino k al azar (k != i)
    vecino_idx = indice_actual
    while vecino_idx == indice_actual:
        vecino_idx = random.randint(0, NUM_OBRERAS - 1)
    vecino = poblacion[vecino_idx]

    nueva = Solucion()
    nueva.cantidades = actual.cantidades[:]

    #2. Elegir dimension j al azar
    j = random.randint(0, len(productos_data) - 1)

    #Operaciones para la Formula

    #Obtenemos las variables de la forumla
    x_ij = actual.cantidades[j]     # Valor actual
    x_kj = vecino.cantidades[j]     # Valor del vecino
    r2 = random.uniform(-1, 1)   # Factor aleatorio [-1, 1]

    #Formula: v_ij = x_ij + r2 * (x_ij - x_kj)
    v_ij_calculado = x_ij + r2 * (x_ij - x_kj)

    #--- ADAPTACION A ENTEROS ---
    #Como vendemos productos enteros, redondeamos el resultado matematico
    cambio_redondeado = int(round(v_ij_calculado - x_ij))

    #Proteccion contra estancamiento (si el cambio fue 0 tras redondear)
    if cambio_redondeado == 0:
        cambio_redondeado = random.choice([-1, 1])

    #Aplicamos el cambio
    nuevo_val = x_ij + cambio_redondeado

    #Restricciones de limites (min/max)
    nuevo_val = max(productos_data[j]["min"], min(nuevo_val, productos_data[j]["max"]))

    nueva.cantidades[j] = nuevo_val
    nueva.evaluar()
    return nueva


# Funcion principal
def algoritmo_abeja():
    poblacion = []

    #--- INICIALIZACION ---
    #Creamos las abejas iniciales y calculamos su fitness
    for _ in range(NUM_OBRERAS):
        sol = Solucion()
        sol.inicializar()
        poblacion.append(sol)

    #Identificamos la mejor solucion inicial
    mejor_solucion_global = copy.deepcopy(poblacion[0])
    for sol in poblacion:
        if sol.fitness > mejor_solucion_global.fitness:
            mejor_solucion_global = copy.deepcopy(sol)

    #--- BUCLE PRINCIPAL (Por generaciones) ---
    for it in range(ITERACIONES):
        #FASE 1: OBRERAS (Exploracion local)
        for i in range(NUM_OBRERAS):
            #Genera una solucion vecina
            nueva_sol = generar_vecino(poblacion, i)

            #Seleccion: Si la vecina es mejor, reemplaza a la actual
            if nueva_sol.fitness > poblacion[i].fitness:
                poblacion[i] = nueva_sol
                poblacion[i].intentos = 0 # Reinicia contador de estancamiento
            else:
                poblacion[i].intentos += 1 # Incrementa contador si no mejora

        #FASE 2: OBSERVADORAS (Explotacion de las mejores)
        #Calculamos aptitudes para la ruleta
        aptitudes = [s.fitness for s in poblacion]

        for _ in range(NUM_OBSERVADORAS):
            #Selecciona una solucion base usando la ruleta
            idx = seleccion_ruleta(aptitudes)

            #Genera un vecino alrededor de la seleccionada
            nueva_sol = generar_vecino(poblacion, idx)

            #Seleccion para la observadora
            if nueva_sol.fitness > poblacion[idx].fitness:
                poblacion[idx] = nueva_sol
                poblacion[idx].intentos = 0
                #Actualizamos la aptitud en la lista para futuras selecciones en este ciclo
                aptitudes[idx] = nueva_sol.fitness
            else:
                poblacion[idx].intentos += 1

        #FASE 3: EXPLORADORAS (Reemplazo de soluciones agotadas)
        for i in range(NUM_OBRERAS):
            #Si una solucion ha fallado en mejorar mas veces que el LIMITE
            if poblacion[i].intentos >= LIMITE:
                #La abeja se convierte en exploradora y busca una nueva solucion aleatoria
                nueva_random = Solucion()
                nueva_random.inicializar()
                poblacion[i] = nueva_random
                poblacion[i].intentos = 0

        #--- ACTUALIZAR MEJOR GLOBAL ---
        mejor_actual = max(poblacion, key=lambda s: s.fitness)

        #Guardamos la mejor encontrada
        if mejor_actual.fitness > mejor_solucion_global.fitness:
            mejor_solucion_global = copy.deepcopy(mejor_actual)

        #IMPRESION DE PROGRESO
        estado = "OK" if mejor_solucion_global.peso_total <= CAPACIDAD_MAXIMA else "SOBREPESO"

        #RESULTADOS
        print(
            f"Gen {it + 1:02d}: {estado:<9} | Cromosoma={mejor_solucion_global.cantidades} | Peso={mejor_solucion_global.peso_total:.1f} | ${mejor_solucion_global.valor_total}")

    return mejor_solucion_global



if __name__ == "__main__":
    print("\n" + "=" * 70)
    resultado = algoritmo_abeja()

    print("\n" + "=" * 70)
    print(" RESULTADO FINAL")
    print("=" * 70)

    # Verificacion de validez
    if resultado.peso_total > CAPACIDAD_MAXIMA:
        print("PRECAUCION: El algoritmo termino con una solucion NO VALIDA (excede peso).")
        print("Intenta ejecutarlo nuevamente.")
    else:
        print("Solucion Valida Encontrada")

    print("\nProductos seleccionados:")
    for i, cant in enumerate(resultado.cantidades):
        nombre = productos_data[i]['nombre']
        if cant > 0:
            print(f"- {nombre:20s}: {cant} unidades")

    print("-" * 40)
    print(f"Peso Total:     {resultado.peso_total:.2f} / {CAPACIDAD_MAXIMA}")
    print(f"Beneficio Total: ${resultado.valor_total}")