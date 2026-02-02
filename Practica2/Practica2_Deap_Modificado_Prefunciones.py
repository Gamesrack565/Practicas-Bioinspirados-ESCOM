#@Angel Abraham Higuera Pineda

#Importa la biblioteca para la generacion de numeros aleatorios.
import random
#Importa la biblioteca para usar operadores estandar como funciones (suma, resta, etc.).
import operator
#Importa NumPy para operaciones numericas eficientes, con el alias 'np'.
import numpy as np
#Importa functools para trabajar con funciones de orden superior, como 'partial'.
import functools
#Importa SymPy para matematicas simbolicas, como simplificar expresiones.
import sympy as sp
#Importa la biblioteca para usar multiples procesos y acelerar el calculo.
#FUNDAMENTAL, YA QUE SIN ELLA EL PROGRAMA TARDA DEMASIADO
import multiprocessing

#Importa las herramientas necesarias de la biblioteca DEAP para programacion genetica.
from deap import algorithms, base, creator, tools, gp


# --- Funciones protegidas ---
#Define una funcion de division protegida para evitar errores de division por cero.
def protected_div(left, right):
    #Suprime temporalmente las advertencias de NumPy sobre division por cero o valores invalidos.
    with np.errstate(divide='ignore', invalid='ignore'):
        #Realiza la division.
        x = np.divide(left, right)
    #Si el resultado es infinito o no es un numero (NaN).
    if np.isinf(x) or np.isnan(x):
        #Devuelve un valor por defecto (1.0).
        return 1.0
    #Si el resultado es valido, lo devuelve.
    return x

#Define una funcion protegida para elevar al cuadrado.
def protected_pow2(x):
    #Suprime temporalmente las advertencias sobre desbordamiento (overflow).
    with np.errstate(over='ignore', invalid='ignore'):
        #Calcula la potencia.
        val = np.power(x, 2)
    #Si el resultado es infinito o NaN.
    if np.isinf(val) or np.isnan(val):
        #Devuelve un valor por defecto (0.0).
        return 0.0
    #Si es valido, devuelve el valor.
    return val

#Define una funcion protegida para elevar al cubo.
def protected_pow3(x):
    #Suprime temporalmente las advertencias sobre desbordamiento.
    with np.errstate(over='ignore', invalid='ignore'):
        #Calcula la potencia.
        val = np.power(x, 3)
    #Si el resultado es infinito o NaN.
    if np.isinf(val) or np.isnan(val):
        #Devuelve un valor por defecto (0.0).
        return 0.0
    #Si es valido, devuelve el valor.
    return val


# --- Primitive set ---
#Crea un conjunto de primitivas (bloques de construccion) llamado "MAIN" que acepta 2 argumentos.
pset = gp.PrimitiveSet("MAIN", 2)
#Renombra los argumentos de entrada a 'x' e 'y' para mayor claridad.
pset.renameArguments(ARG0='x', ARG1='y')
#Anade la operacion de suma (2 argumentos) al conjunto.
pset.addPrimitive(operator.add, 2)
#Anade la operacion de resta (2 argumentos).
pset.addPrimitive(operator.sub, 2)
#Anade la operacion de multiplicacion (2 argumentos).
pset.addPrimitive(operator.mul, 2)
#Anade la funcion de division protegida (2 argumentos).
pset.addPrimitive(protected_div, 2)
#Anade la operacion de negacion (1 argumento).
pset.addPrimitive(operator.neg, 1)
#Anade la funcion de potencia al cuadrado protegida (1 argumento).
pset.addPrimitive(protected_pow2, 1)
#Anade la funcion de potencia al cubo protegida (1 argumento).
pset.addPrimitive(protected_pow3, 1)
#Anade una constante efimera que genera un numero aleatorio entre -2.0 y 2.0.
pset.addEphemeralConstant("rand_small", functools.partial(random.uniform, -2.0, 2.0))


# --- DEAP setup ---
#Crea una definicion de 'Fitness' para minimizacion (un solo objetivo, menor es mejor).
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#Crea la estructura de un 'Individuo', que es un arbol de primitivas con el 'Fitness' definido.
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

#Crea una 'Toolbox' para almacenar las herramientas y parametros de la evolucion.
toolbox = base.Toolbox()
#Registra una herramienta para generar expresiones (arboles) usando el metodo 'HalfAndHalf'.
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
#Registra una herramienta para crear un individuo completo a partir de una expresion.
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
#Registra una herramienta para crear una poblacion como una lista de individuos.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#Registra una herramienta para compilar un arbol de expresion en una funcion de Python ejecutable.
toolbox.register("compile", gp.compile, pset=pset)

# --- Datos de entrenamiento ---
#Genera 500 puntos de entrenamiento (x, y) con valores aleatorios entre -1.0 y 1.0.
train_points = np.random.uniform(-1.0, 1.0, (500, 2))
#Separa las columnas en arreglos 'xs' e 'ys'.
xs, ys = train_points[:, 0], train_points[:, 1]
#Calcula los valores objetivo usando la funcion real que queremos que el sistema descubra.
target_raw = 5 * xs ** 3 * ys ** 2 + xs / 2.0


# --- Evaluación ---
#Define la funcion para evaluar la aptitud (fitness) de un individuo.
def eval_symb_reg(individual):
    #Compila el arbol del individuo en una funcion de Python.
    func = toolbox.compile(expr=individual)
    #Inicia un bloque try-except para manejar posibles errores durante la evaluacion.
    try:
        #Aplica la funcion a cada punto de entrenamiento y obtiene las predicciones.
        preds = np.fromiter((func(x, y) for x, y in train_points), dtype=np.float64, count=len(train_points))
        #Si alguna prediccion es NaN o infinita, la solucion no es valida.
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            #Devuelve una aptitud muy mala (un numero muy grande).
            return (1e12,)
        #Calcula el Error Cuadratico Medio (MSE) entre las predicciones y los valores objetivo.
        mse = np.mean((preds - target_raw) ** 2)
        #Devuelve el MSE como una tupla (formato requerido por DEAP).
        return (mse,)
    #Si ocurre cualquier otra excepcion durante la evaluacion.
    except Exception:
        #Devuelve una aptitud muy mala.
        return (1e12,)


#Define una funcion para convertir un individuo de DEAP a una expresion simbolica de SymPy.
def compile_to_sympy(individual):
    #Crea los simbolos 'x' e 'y' para la expresion matematica.
    x, y = sp.symbols('x y')
    #Define un diccionario para mapear los nombres de las funciones de DEAP a sus equivalentes en SymPy.
    local_dict = {
        'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b,
        'protected_div': lambda a, b: a / b, 'neg': lambda a: -a,
        'protected_pow2': lambda a: a ** 2, 'protected_pow3': lambda a: a ** 3,
        'x': x, 'y': y
    }
    #Convierte el arbol del individuo a una cadena de texto (ej. "add(mul(x, y), 2.5)").
    code = str(individual)
    #Busca todas las constantes efimeras en el individuo.
    ephemerals = [p for p in individual if isinstance(p, gp.Terminal) and p.name.startswith("rand")]
    #Reemplaza los nombres de las constantes (ej. "rand_small") por sus valores numericos en la cadena.
    for eph in reversed(ephemerals):
        code = code.replace(eph.name, f"({eph.value})")
    #Inicia un bloque try-except para manejar errores en la conversion.
    try:
        #Evalua de forma segura la cadena de texto para construir una expresion de SymPy.
        expr = eval(code, {"__builtins__": None}, local_dict)
        #Simplifica la expresion matematica resultante y la devuelve.
        return sp.simplify(expr)
    #Si ocurre un error durante la simplificacion.
    except Exception as e:
        #Devuelve un mensaje de error.
        return f"Error al simplificar: {e}"


# --- Bloque principal seguro ---
#Asegura que el codigo dentro de este bloque solo se ejecute si el script es el principal.
if __name__ == "__main__":
    #Imprime un mensaje de inicio.
    print("Evolución rápida iniciada...")

    #Fijar semilla opcionalmente para resultados reproducibles (actualmente comentado).
    #random.seed(42)
    #np.random.seed(42)

    #Crea un 'pool' de procesos para usar multiples nucleos del CPU.
    pool = multiprocessing.Pool()
    #Registra la funcion 'map' del pool en la toolbox para que DEAP la use.
    toolbox.register("map", pool.map)

    #Registra la funcion de evaluacion en la toolbox.
    toolbox.register("evaluate", eval_symb_reg)
    #Registra el metodo de seleccion (torneo de tamano 7).
    toolbox.register("select", tools.selTournament, tournsize=7)
    #Registra el metodo de cruce (crossover de un punto).
    toolbox.register("mate", gp.cxOnePoint)
    #Registra un metodo para generar expresiones pequenas para la mutacion.
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    #Registra el metodo de mutacion (mutacion uniforme, que reemplaza un subarbol).
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    #Anade un "decorador" para limitar la altura maxima de los arboles a 17 despues del cruce.
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    #Anade un "decorador" para limitar la altura maxima de los arboles a 17 despues de la mutacion.
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    #Crea la poblacion inicial con 1500 individuos.
    pop = toolbox.population(n=1500)
    #Crea un objeto 'Hall of Fame' para guardar al mejor individuo encontrado.
    hof = tools.HallOfFame(1)

    #Configura la recoleccion de estadisticas sobre la aptitud (fitness).
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    #Configura la recoleccion de estadisticas sobre el tamano (numero de nodos) de los individuos.
    stats_size = tools.Statistics(len)
    #Combina ambas estadisticas en un solo objeto.
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    #Registra las metricas a calcular: promedio, desviacion estandar y minimo.
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)

    # --- Versión modificada de eaSimple con impresión de la mejor función ---
    #Define el numero de generaciones que durara la evolucion.
    NGEN = 110
    #Define la probabilidad de cruce y de mutacion.
    CXPB, MUTPB = 0.7, 0.2

    #Imprime un mensaje para indicar que el proceso evolutivo comienza.
    print("\nIniciando evolución...")

    #Inicia el bucle principal que se ejecutara para cada generacion.
    for gen in range(1, NGEN + 1):
        # --- Selección y reproducción ---
        #Selecciona los individuos de la poblacion actual para formar la siguiente generacion.
        offspring = toolbox.select(pop, len(pop))
        #Crea clones de los individuos seleccionados para no modificar los originales.
        offspring = list(map(toolbox.clone, offspring))

        #Aplica el operador de cruce (crossover) a la descendencia.
        #Itera sobre la descendencia en pares (hijo1, hijo2).
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            #Si un numero aleatorio es menor que la probabilidad de cruce.
            if random.random() < CXPB:
                #Realiza el cruce entre los dos individuos.
                toolbox.mate(child1, child2)
                #Elimina la aptitud de los hijos, ya que su material genetico cambio.
                del child1.fitness.values
                del child2.fitness.values

        #Aplica el operador de mutacion a la descendencia.
        #Itera sobre cada individuo en la descendencia.
        for mutant in offspring:
            #Si un numero aleatorio es menor que la probabilidad de mutacion.
            if random.random() < MUTPB:
                #Aplica la mutacion al individuo.
                toolbox.mutate(mutant)
                #Elimina la aptitud del mutante, ya que ha cambiado.
                del mutant.fitness.values

        #Evalua los individuos cuya aptitud no es valida.
        #Crea una lista de todos los individuos que fueron modificados (sin aptitud valida).
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #Calcula la aptitud para todos los individuos invalidos, usando multiprocesamiento.
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        #Asigna la nueva aptitud calculada a cada individuo correspondiente.
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        #Reemplaza la poblacion antigua con la nueva descendencia ya evaluada.
        pop[:] = offspring

        #Actualiza el Hall of Fame con los mejores individuos de la nueva poblacion.
        hof.update(pop)
        #Calcula las estadisticas para la generacion actual.
        record = mstats.compile(pop)

        #Obtiene el mejor individuo de la poblacion actual para imprimirlo.
        best_ind = tools.selBest(pop, 1)[0]

        # --- Impresión estándar + mejor función encontrada ---
        #Imprime el resumen estadistico de la generacion.
        print(f"Gen {gen:03d} | Fitness promedio = {record['fitness']['avg']:.4e} | "
              f"Mejor = {record['fitness']['min']:.4e} | Tamaño promedio = {record['size']['avg']:.2f}")
        #Imprime la expresion del mejor individuo encontrado en esta generacion.
        print(f"  ↳ Mejor función encontrada: {best_ind}\n")

    # --- Fin del ciclo evolutivo ---
    #Imprime un mensaje indicando que la evolucion ha terminado.
    print("Evolución finalizada")

    #Cierra el pool de procesos para liberar los recursos.
    pool.close()
    #Espera a que todos los procesos del pool terminen completamente.
    pool.join()

    #Muestra el mejor individuo encontrado en toda la evolucion.
    #Obtiene el mejor individuo guardado en el Hall of Fame.
    best = hof[0]
    #Imprime la cabecera de los resultados finales.
    print("\n--- Mejor Individuo Encontrado ---")
    #Imprime la aptitud final (Error Cuadratico Medio) del mejor individuo.
    print(f"Fitness (MSE Real): {best.fitness.values[0]}")
    #Imprime otra cabecera.
    print("\n--- Expresión Simplificada ---")
    #Convierte la expresion del mejor individuo a un formato matematico simplificado y la imprime.
    print(compile_to_sympy(best))