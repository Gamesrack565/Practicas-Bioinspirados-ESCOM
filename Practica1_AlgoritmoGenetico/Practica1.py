import random
from typing import List, Tuple

#Datos del problema a trabajar
#Incluye nombre, peso y precio de cada producto
productos = [
    {'nombre': 'Decoy Detonators', 'peso': 4,   'precio': 10},
    {'nombre': 'Fever Fudge',      'peso': 2,   'precio': 3},
    {'nombre': 'Love Potion',      'peso': 2,   'precio': 8},
    {'nombre': 'Puking Pastilles', 'peso': 1.5, 'precio': 2},
    {'nombre': 'Extendable Ears',  'peso': 5,   'precio': 12},
    {'nombre': 'Nosebleed Nougat', 'peso': 1,   'precio': 2},
    {'nombre': 'Skiving Snackbox', 'peso': 5,   'precio': 6},
]

#Variables utilizadas
#Cantidad de productos que hay
NUM_productos = len(productos)
#Cantidad maxima de unidades por producto
MAX_productos = 10
#Peso maximo de la mochila
MAX_capacidad_mochila = 30.0

#Restricciones
MIN_lovep = 3       #Indice 2
MIN_skiving = 2     #Indice 6

#Parameros del algoritmo genetico
#Tamano de la poblacion
TAM_poblacion = 10
#Numero de generaciones
Generaciones = 50
#Probabilidad de cruza
Proba_cruza = 0.85
#Probabilidad de mutacion
Proba_muta = 0.1

#Funciones necesarias
#Calcula el peso total del cromosoma
def peso_total(cromosoma: List[int]) -> float:
    return sum(q * it["peso"] for q, it in zip(cromosoma, productos))
    #zip empareja cada cantidad con el diccionario del producto correspondiente

#Verifica que el cromosoma cumpla las restricciones (rango, mínimos y capacidad)
def es_valido(cromosoma: List[int]) -> bool:
    #any comprueba si existe algún valor fuera del rango [0, MAX_UNITS]
    if any(q < 0 or q > MAX_productos for q in cromosoma):
        return False
    #Comprueba si cumple los minimos obligatorios
    if cromosoma[2] < MIN_lovep or cromosoma[6] < MIN_skiving:
        return False
    #Comprueba si no ha excedido la capacidad maxima
    if peso_total(cromosoma) > MAX_capacidad_mochila:
        return False
    return True

#La funcion fitness devuelve el beneficio total del cromosoma si es valido, 0 en caso contrario
def fitness(cromosoma: List[int]) -> float:
    #Revisamos si el cromosoma es valido
    #Si no lo es, devolvemos 0
    if not es_valido(cromosoma):
        return 0
    #Si es valido, devolvemos la suma del beneficio total
    return sum(q * it["precio"] for q, it in zip(cromosoma, productos))

#Repara un cromosoma:
#fuerza límites enteros [0, MAX_UNITS],
#fuerza los mínimos obligatorios (Love y Skiving),
#reduce cantidades (excepto mínimos) hasta que el peso ≤ CAPACIDAD_MAX.
def reparar(cromosoma: List[int]) -> List[int]:
    #Convertir a enteros y asegurar límites por seguridad
    c = [max(0, min(MAX_productos, int(round(x)))) for x in cromosoma]

    #Forzar mínimos
    c[2] = max(c[2], MIN_lovep)
    c[6] = max(c[6], MIN_skiving)

    #Mientras sobrepase la capacidad, reducimos unidades de productos "no obligatorios"
    while peso_total(c) > MAX_capacidad_mochila:
        #Candidatos: índices distintos a 2 y 6 con cantidad > 0
        idxs = [i for i in range(NUM_productos) if i not in (2, 6) and c[i] > 0]
        if not idxs:
            #Si no hay candidatos, salimos (no podemos reparar más sin violar mínimos)
            break
        i = random.choice(idxs)   #Elegimos uno al azar para reducir (diversidad)
        c[i] -= 1
    return c

#Creacion de cromosomas e inicializacion de la poblacion
def creacion_individuo() -> List[int]:
    #Empieza con 0 unidades para cada producto
    crom = [0] * NUM_productos     
    peso = 0.0

    #Forzamos a que esten los minimos obligatorios en la poscion del cromosoma
    crom[2] = MIN_lovep
    peso += productos[2]['peso'] * MIN_lovep
    crom[6] = MIN_skiving
    peso += productos[6]['peso'] * MIN_skiving

    #Generamos un orden aleatorio de índices y tratamos de meter productos hasta llenar
    posiciones = list(range(NUM_productos))
    random.shuffle(posiciones)    #reordena posiciones para variar la construcción

    #Recorremos cada índice en posiciones
    for i in posiciones:
        #Si al menos una unidad entra sin superar capacidad
        if peso + productos[i]['peso'] <= MAX_capacidad_mochila:
            #Maximo que cabe por espacio restante (división entera)
            max_cabe = int((MAX_capacidad_mochila - peso) // productos[i]['peso'])
            disponible = MAX_productos - crom[i]            # cuántos hay disponibles aún
            limite = min(max_cabe, disponible)         # límite real que podemos añadir
            if limite > 0:
                cantidad = random.randint(1, limite)   # elegimos aleatoriamente 1..limite
                crom[i] += cantidad
                peso += productos[i]['peso'] * cantidad
    return crom

#Crea la población inicial llamando a creacion_individuo 'size' veces.
def init_population(size=TAM_poblacion) -> List[List[int]]:
    return [creacion_individuo() for _ in range(size)]
    #la sintaxis [f() for _ in range(size)] crea 'size' individuos

#Ruletajhkhjk
def seleccion_ruleta(poblacion: List[List[int]], aptitudes: List[float]) -> int:
    total = sum(aptitudes)
    if total <= 0:
        #Si todas las aptitudes son cero o negativas, devolvemos un índice aleatorio
        return random.randrange(len(poblacion))
    r = random.uniform(0, total)   #Número aleatorio entre 0 y total
    acumulado = 0.0
    #Recorrer cada individuo acumulando aptitud hasta pasar r
    for i, f in enumerate(aptitudes):
        acumulado += f
        if acumulado >= r:
            return i
    return len(poblacion) - 1

#Cruza con la probabilidad 0.5 por gen
def cruza_uniforme(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    h1, h2 = [], []
    for g1, g2 in zip(p1, p2):     #zip recorre gen a gen
        if random.random() < 0.5:
            h1.append(g1); h2.append(g2)
        else:
            h1.append(g2); h2.append(g1)
    return h1, h2

#Mutar
#Aplica mutación a un cromosoma con probabilidad p.
#Hay dos tipos de mutación:
#Reset: asigna a un gen un valor aleatorio entre 0 y MAX_UNITS.
#Incremento/Decremento: suma o resta 1 al valor de un gen (respetando límites).
#Después de mutar, repara el cromosoma para garantizar que cumple restricciones.
def mutar(crom: List[int], p: float) -> List[int]:
    c = crom[:]
    for i in range(NUM_productos):
        if random.random() < p:  # probabilidad por gen
            if random.random() < 0.5:
                # Reset
                c[i] = random.randint(0, MAX_productos)
            else:
                # Incremento o decremento
                c[i] = max(0, min(MAX_productos, c[i] + random.choice([-1, 1])))
    return reparar(c)



#Padre más débil
def reemplazar_debil(poblacion: List[List[int]], aptitudes: List[float], hijo: List[int], apt_hijo: float):
    i_min = min(range(len(aptitudes)), key=lambda i: aptitudes[i])
    if apt_hijo > aptitudes[i_min]:
        poblacion[i_min] = hijo
        aptitudes[i_min] = apt_hijo

#FUNCION PRINCIPAL
def algoritmo_genetico():
    #población inicial
    poblacion = init_population(TAM_poblacion)     
    aptitudes = [fitness(ind) for ind in poblacion]   #lista de fitness para cada individuo

    #Ciclo principal: por cada generación
    for gen in range(1, Generaciones + 1):
        #Arreglos de probabilidades
        prob_cruza = [random.random() for _ in range(TAM_poblacion // 2)]
        prob_mut = [random.random() for _ in range(TAM_poblacion)]

        #Por cada par de hijos a crear
        for i in range(TAM_poblacion // 2):
            #Seleccionar padres por ruleta
            p1 = seleccion_ruleta(poblacion, aptitudes)
            p2 = seleccion_ruleta(poblacion, aptitudes)
            #Crear copias (hijos temporales)
            h1, h2 = poblacion[p1][:], poblacion[p2][:]

            #Decidir si cruzan los padres
            if prob_cruza[i] < Proba_cruza:
                h1, h2 = cruza_uniforme(poblacion[p1], poblacion[p2])

            #Mutar cada hijo usando probabilidades del arreglo
            h1 = mutar(h1, prob_mut[2*i] if 2*i < TAM_poblacion else Proba_muta)
            h2 = mutar(h2, prob_mut[2*i+1] if 2*i+1 < TAM_poblacion else Proba_muta)

            #Evaluar y reemplazar si son mejores que el padre más débil
            a1, a2 = fitness(h1), fitness(h2)
            reemplazar_debil(poblacion, aptitudes, h1, a1)
            reemplazar_debil(poblacion, aptitudes, h2, a2)

        #Mostrar las generaciones
        best_idx = max(range(len(poblacion)), key=lambda i: aptitudes[i])
        print(f"Gen {gen:2d}: mejor fitness = {aptitudes[best_idx]:.2f}, "
                  f"crom = {poblacion[best_idx]}, peso = {peso_total(poblacion[best_idx]):.2f}")

    #Devolver mejor solución encontrada
    best_idx = max(range(len(poblacion)), key=lambda i: aptitudes[i])
    return poblacion[best_idx], aptitudes[best_idx], peso_total(poblacion[best_idx])


if __name__ == "__main__":
    mejor, apt, peso = algoritmo_genetico()
    print("\nMejor solución encontrada:")
    for qty, item in zip(mejor, productos):
        print(f"- {item['nombre']}: {qty}")
    print(f"Peso total = {peso:.2f} / {MAX_capacidad_mochila}, Beneficio = {apt}")
