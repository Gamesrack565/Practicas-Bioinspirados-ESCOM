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
#Crea un conjunto de primitivas llamado "MAIN" que acepta 2 argumentos.
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


# --- Princiapl
if __name__ == "__main__":
    print("///  PROGRAMACION GENETICA    ///")
    print("By: Abraham Higuera")
    print("----------------------------------------------")
    print("Bienvenido a mi programa\n")
    print("Funcion a buscar: 5 * xs ** 3 * ys ** 2 + xs / 2.0")

    #Semilla dada para pruebas, no es necesaria
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
    pop = toolbox.population(n=1000)
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

    #Ejecuta el algoritmo evolutivo 'eaSimple' con los parametros definidos.
    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100,
        stats=mstats, halloffame=hof, verbose=True
    )

    #Cierra el pool de procesos de forma segura.
    pool.close()
    #Espera a que todos los procesos terminen.
    pool.join()

    #Obtiene el mejor individuo del 'Hall of Fame'.
    best = hof[0]
    #Imprime la cabecera de los resultados.
    print("\n--- Mejor Individuo Encontrado ---")
    #Imprime la aptitud (MSE) del mejor individuo.
    print(f"Fitness (MSE Real): {best.fitness.values[0]}")
    #Imprime otra cabecera.
    print("\n--- Expresion Simplificada ---")
    #Llama a la funcion para simplificar e imprimir la expresion matematica encontrada.
    print(compile_to_sympy(best))