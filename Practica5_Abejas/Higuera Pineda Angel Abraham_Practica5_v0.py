# Angel Abraham Higuera Pineda
# Practica: Abejas
# Profesora: Abril Valeria Uriarte Arcia

# Importa la libreria para generar numeros aleatorios (esencial para algoritmos estocasticos).
import random
# Importa 'List' para el tipado estatico en las funciones (buena practica).
from typing import List

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
# Define el peso maximo que puede soportar la mochila (restriccion fuerte).
CAPACIDAD_MAXIMA = 30

# Define la base de datos de los productos disponibles.
# Cada diccionario representa un producto con sus caracteristicas y restricciones individuales.
# [Nombre, Peso, Precio, Min, Max]
productos_data = [
    {"nombre": "Decoy Detonators", "peso": 4, "precio": 10, "min": 0, "max": 10},
    {"nombre": "Fever Fudge", "peso": 2, "precio": 3, "min": 0, "max": 10},
    # Nota: Love Potion tiene un minimo de 3 unidades obligatorias.
    {"nombre": "Love Potion", "peso": 2, "precio": 8, "min": 3, "max": 10},
    {"nombre": "Puking Pastilles", "peso": 1.5, "precio": 2, "min": 0, "max": 10},
    {"nombre": "Extendable Ears", "peso": 5, "precio": 12, "min": 0, "max": 10},
    {"nombre": "Nosebleed Nougat", "peso": 1, "precio": 2, "min": 0, "max": 10},
    # Nota: Skiving Snackbox tiene un minimo de 2 unidades obligatorias.
    {"nombre": "Skiving Snackbox", "peso": 5, "precio": 6, "min": 2, "max": 10}
]

# Parámetros del algoritmo ABC
# Numero total de fuentes de alimento (soluciones) en memoria.
TAMANO_ENJAMBRE = 40
# Numero de abejas que explotan soluciones conocidas.
NUM_OBRERAS = 20
# Numero de abejas que eligen soluciones basadas en la danza (fitness) de las obreras.
NUM_OBSERVADORAS = 20
# Numero maximo de intentos fallidos antes de abandonar una solucion (limite de estancamiento).
LIMITE = 5
# Numero total de generaciones o ciclos del algoritmo.
ITERACIONES = 50


# ==============================================================================
# 2. CLASES Y FUNCIONES
# ==============================================================================
# Clase que representa una solucion individual (una combinacion de productos).
class Solucion:
    def __init__(self):
        # Lista que guardara la cantidad de cada producto.
        self.cantidades = []
        # Valor de aptitud de la solucion (que tan buena es).
        self.fitness = 0
        # Peso acumulado de los productos seleccionados.
        self.peso_total = 0
        # Valor monetario acumulado de los productos.
        self.valor_total = 0
        # Contador de intentos fallidos para mejorar esta solucion (para la fase de exploradoras).
        self.intentos = 0

    # Metodo para crear una solucion inicial valida.
    def inicializar(self):
        # Inicialización inteligente para encontrar soluciones válidas rápido.
        # Primero, aseguramos cumplir con los minimos requeridos de cada producto.
        self.cantidades = [p["min"] for p in productos_data]

        # Creamos una lista de indices y los mezclamos para llenar la mochila en orden aleatorio.
        indices = list(range(len(productos_data)))
        random.shuffle(indices)

        # Iteramos sobre los productos en orden aleatorio para intentar agregar mas.
        for i in indices:
            # Mientras no superemos el maximo permitido para este producto.
            while self.cantidades[i] < productos_data[i]["max"]:
                # Calculamos el peso actual de la mochila.
                peso_actual = sum(self.cantidades[k] * productos_data[k]["peso"] for k in range(len(productos_data)))
                # Obtenemos el peso de agregar una unidad mas del producto actual.
                peso_extra = productos_data[i]["peso"]

                # Si agregar el producto no excede la capacidad maxima de la mochila.
                if peso_actual + peso_extra <= CAPACIDAD_MAXIMA:
                    # Agregamos una unidad.
                    self.cantidades[i] += 1
                else:
                    # Si ya no cabe, dejamos de intentar agregar este producto.
                    break
        # Calculamos el fitness inicial de la solucion generada.
        self.evaluar()

    # Metodo para calcular el fitness (calidad) de la solucion.
    def evaluar(self):
        self.peso_total = 0
        self.valor_total = 0
        # Recorremos todos los productos para sumar pesos y precios.
        for i, p in enumerate(productos_data):
            self.peso_total += self.cantidades[i] * p["peso"]
            self.valor_total += self.cantidades[i] * p["precio"]

        # Validamos la restriccion de peso.
        if self.peso_total <= CAPACIDAD_MAXIMA:
            # Si es valido, el fitness es el valor monetario total.
            self.fitness = self.valor_total
        else:
            # Si se pasa del peso, penalizamos fuertemente el fitness (casi cero).
            self.fitness = 1e-6


# Funcion para seleccionar un indice basado en probabilidad (Ruleta).
# Las soluciones con mayor fitness tienen mas probabilidad de ser elegidas.
def seleccion_ruleta(aptitudes: List[float]) -> int:
    # Suma total de aptitudes.
    total = sum(aptitudes)
    # Proteccion contra division por cero o aptitudes nulas.
    if total <= 0: return random.randrange(len(aptitudes))

    # Generamos un punto aleatorio en la ruleta.
    r = random.uniform(0, total)
    acumulado = 0.0
    # Recorremos acumulando probabilidades hasta alcanzar el punto aleatorio.
    for i, f in enumerate(aptitudes):
        acumulado += f
        if acumulado >= r: return i
    # Retorno por defecto (ultimo elemento) en caso de errores de redondeo.
    return len(aptitudes) - 1


# Funcion central de ABC: Modifica una solucion existente basandose en otra vecina.
# Formula: v_ij = x_ij + phi * (x_ij - x_kj)
def generar_vecino(poblacion, indice_actual):
    # Obtenemos la solucion actual (x_i).
    actual = poblacion[indice_actual]

    # Elegimos un vecino aleatorio (x_k) diferente al actual.
    vecino_idx = indice_actual
    while vecino_idx == indice_actual:
        vecino_idx = random.randint(0, NUM_OBRERAS - 1)
    vecino = poblacion[vecino_idx]

    # Creamos una nueva instancia para la solucion candidata.
    nueva = Solucion()
    # Copiamos las cantidades de la solucion original.
    nueva.cantidades = actual.cantidades[:]

    # Elegimos una dimension aleatoria (j) para modificar (un producto especifico).
    j = random.randint(0, len(productos_data) - 1)
    # Generamos el factor de cambio aleatorio (phi) entre -1 y 1.
    phi = random.uniform(-1, 1)

    # Aplicamos la ecuacion de ABC.
    nuevo_val = actual.cantidades[j] + phi * (actual.cantidades[j] - vecino.cantidades[j])

    # Como es un problema discreto (cantidades enteras), redondeamos.
    nuevo_val = int(round(nuevo_val))

    # Aseguramos que el nuevo valor respete los limites min y max del producto.
    nuevo_val = max(productos_data[j]["min"], min(nuevo_val, productos_data[j]["max"]))

    # Asignamos el nuevo valor y evaluamos la nueva solucion candidata.
    nueva.cantidades[j] = nuevo_val
    nueva.evaluar()
    return nueva


# Funcion auxiliar para clonar una solucion (Deep Copy manual).
def copiar_solucion(origen):
    dest = Solucion()
    dest.cantidades = origen.cantidades[:]
    dest.fitness = origen.fitness
    dest.peso_total = origen.peso_total
    dest.valor_total = origen.valor_total
    dest.intentos = origen.intentos
    return dest


# ==============================================================================
# 3. ALGORITMO PRINCIPAL
# ==============================================================================
def algoritmo_abc():
    poblacion = []
    # --- Inicialización ---
    # Creamos la poblacion inicial de abejas obreras.
    for _ in range(NUM_OBRERAS):
        sol = Solucion()
        sol.inicializar()  # Llama a la generacion inteligente.
        poblacion.append(sol)

    # Identificamos la mejor solucion inicial para guardarla.
    mejor_solucion_global = max(poblacion, key=lambda s: s.fitness)
    # Hacemos una copia independiente de la mejor solucion.
    mejor_global_copia = copiar_solucion(mejor_solucion_global)

    # --- Bucle Principal de Iteraciones ---
    for it in range(ITERACIONES):

        # --- FASE 1: OBRERAS (Employee Bees) ---
        # Cada obrera va a una fuente de comida y busca una variante vecina.
        for i in range(NUM_OBRERAS):
            # Genera una solucion vecina usando la ecuacion de ABC.
            nueva_sol = generar_vecino(poblacion, i)

            # Seleccion voraz (Greedy): Si la nueva es mejor, reemplaza a la vieja.
            if nueva_sol.fitness > poblacion[i].fitness:
                poblacion[i] = nueva_sol
                # Reseteamos el contador de intentos fallidos.
                poblacion[i].intentos = 0
            else:
                # Si no mejora, incrementamos el contador de estancamiento.
                poblacion[i].intentos += 1

        # --- FASE 2: OBSERVADORAS (Onlooker Bees) ---
        # Calculamos todas las aptitudes para la ruleta.
        lista_aptitudes = [s.fitness for s in poblacion]

        # Las observadoras eligen soluciones para explotar basandose en probabilidad.
        for _ in range(NUM_OBSERVADORAS):
            # Seleccionamos una solucion 'padre' usando la ruleta.
            idx_seleccionado = seleccion_ruleta(lista_aptitudes)

            # Generamos un vecino alrededor de la solucion seleccionada.
            nueva_sol = generar_vecino(poblacion, idx_seleccionado)

            # Seleccion voraz nuevamente sobre la solucion seleccionada.
            if nueva_sol.fitness > poblacion[idx_seleccionado].fitness:
                poblacion[idx_seleccionado] = nueva_sol
                poblacion[idx_seleccionado].intentos = 0
                # Actualizamos la lista de aptitudes para las siguientes observadoras en este ciclo.
                lista_aptitudes[idx_seleccionado] = nueva_sol.fitness
            else:
                poblacion[idx_seleccionado].intentos += 1

        # --- FASE 3: EXPLORADORAS (Scout Bees) ---
        # Buscamos soluciones estancadas que superen el LIMITE de intentos.
        for i in range(NUM_OBRERAS):
            if poblacion[i].intentos >= LIMITE:
                # Si se supera el limite, la abeja abandona la solucion y busca una nueva aleatoria.
                poblacion[i] = Solucion()
                poblacion[i].inicializar()

        # --- Actualizar Mejor Global ---
        # Buscamos la mejor de la poblacion actual.
        mejor_actual = max(poblacion, key=lambda s: s.fitness)
        # Si supera a la mejor historica, actualizamos la historica.
        if mejor_actual.fitness > mejor_global_copia.fitness:
            mejor_global_copia = copiar_solucion(mejor_actual)

        # --- IMPRESIÓN CON EL FORMATO EXACTO SOLICITADO ---
        # Imprime el progreso de la iteracion.
        print(
            f"Gen {it + 1:2d}: mejor fitness = {mejor_global_copia.fitness:.2f}, crom = {mejor_global_copia.cantidades}, peso = {mejor_global_copia.peso_total:.2f}")

    # Devolvemos la mejor solucion encontrada en todas las iteraciones.
    return mejor_global_copia


# ==============================================================================
# 4. EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    # Ejecutamos el algoritmo.
    resultado = algoritmo_abc()

    print("\nMejor solución encontrada:")
    # Mostramos los detalles de los productos seleccionados.
    for i, cant in enumerate(resultado.cantidades):
        print(f"- {productos_data[i]['nombre']}: {cant}")

    # Formato final exacto solicitado: Peso y Beneficio total.
    print(f"Peso total = {resultado.peso_total:.2f} / {float(CAPACIDAD_MAXIMA)}, Beneficio = {int(resultado.fitness)}")