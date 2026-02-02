#@Angel Abraham Higuera Pineda

#Importa la biblioteca 'math' para operaciones matematicas como seno y coseno.
import math
#Importa la biblioteca 'operator' para usar operadores estandar (+, -, *) como funciones.
import operator
#Importa la biblioteca 'numpy' para calculos numericos eficientes.
import numpy

#Importa los componentes necesarios de la biblioteca DEAP para programacion genetica.
from deap import algorithms, base, creator, tools, gp


# --- 1. Definición del Conjunto de Primitivas (lo que tenías) ---
#Define una funcion de division protegida para evitar errores por division entre cero.
def protectedDiv(left, right):
    #Inicia un bloque 'try' para intentar ejecutar un codigo que puede fallar.
    try:
        #Devuelve el resultado de la division si es exitosa.
        return left / right
    #Si se produce un error especifico de division por cero.
    except ZeroDivisionError:
        #Devuelve un valor numerico por defecto (1) para evitar que el programa se detenga.
        return 1


#Crea el conjunto de primitivas (pset) para una funcion con 1 variable de entrada (ARG0).
pset = gp.PrimitiveSet("MAIN", 1)
#Anade la operacion de suma (requiere 2 argumentos) al conjunto.
pset.addPrimitive(operator.add, 2)
#Anade la operacion de resta (2 argumentos).
pset.addPrimitive(operator.sub, 2)
#Anade la operacion de multiplicacion (2 argumentos).
pset.addPrimitive(operator.mul, 2)
#Anade la funcion de division protegida definida anteriormente (2 argumentos).
pset.addPrimitive(protectedDiv, 2)
#Anade la operacion de negacion (cambio de signo, 1 argumento).
pset.addPrimitive(operator.neg, 1)
#Anade la funcion coseno (1 argumento).
pset.addPrimitive(math.cos, 1)
#Anade la funcion seno (1 argumento).
pset.addPrimitive(math.sin, 1)
#Renombra el argumento de entrada de 'ARG0' a 'x' para mayor claridad.
pset.renameArguments(ARG0='x')

# --- 2. Creación de Tipos (lo que faltaba) ---
#Se crea la funcion de fitness (aptitud) para minimizacion, con un solo objetivo (menor es mejor).
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#Se crea la estructura del 'Individuo' como un arbol de primitivas, con el fitness definido antes.
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# --- 3. Inicialización de Herramientas (Toolbox) ---
#Crea una 'Toolbox' para almacenar las herramientas y parametros del algoritmo genetico.
toolbox = base.Toolbox()
#Registra una herramienta para generar expresiones (arboles) usando el metodo 'HalfAndHalf'.
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
#Registra una herramienta para crear un individuo completo a partir de una expresion.
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
#Registra una herramienta para crear una poblacion como una lista de individuos.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#Registra una herramienta para compilar un arbol de expresion en una funcion de Python que se puede llamar.
toolbox.register("compile", gp.compile, pset=pset)


#Define la funcion de evaluacion: calcula el error cuadratico medio.
def evalSymbReg(individual, points):
    #Compila el arbol del individuo para convertirlo en una funcion ejecutable.
    func = toolbox.compile(expr=individual)
    #La funcion objetivo del ejemplo es: x**4 + x**3 + x**2 + x
    #Calcula los errores al cuadrado entre la funcion del individuo y la funcion objetivo para cada punto.
    sqerrors = ((func(x) - (x ** 4 + x ** 3 + x ** 2 + x)) ** 2 for x in points)
    #Devuelve el Error Cuadratico Medio (MSE) como una tupla (formato requerido por DEAP).
    return math.fsum(sqerrors) / len(points),


#Registro de las herramientas geneticas en la toolbox.
#Registra la funcion de evaluacion, pasandole la lista de puntos de prueba.
toolbox.register("evaluate", evalSymbReg, points=[x / 10. for x in range(-10, 10)])
#Registra el metodo de seleccion (torneo de tamano 3).
toolbox.register("select", tools.selTournament, tournsize=3)
#Registra el metodo de cruce (crossover de un punto).
toolbox.register("mate", gp.cxOnePoint)
#Registra un metodo para generar expresiones pequenas para usar en la mutacion.
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
#Registra el metodo de mutacion (mutacion uniforme, que reemplaza un subarbol).
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

#Se definen limites para la altura de los arboles para evitar que crezcan demasiado (bloating).
#Anade un "decorador" para limitar la altura maxima de los arboles a 17 despues del cruce.
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
#Anade un "decorador" para limitar la altura maxima de los arboles a 17 despues de la mutacion.
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


#Define la funcion principal que ejecutara el proceso evolutivo.
def main():
    #Fija una semilla para que los resultados aleatorios sean reproducibles (actualmente comentado).
    #random.seed(318)

    #Crea la poblacion inicial con 300 individuos.
    pop = toolbox.population(n=300)
    #Crea un objeto 'Hall of Fame' para almacenar al mejor individuo encontrado.
    hof = tools.HallOfFame(1)

    #Configura la recoleccion de estadisticas sobre la aptitud (fitness).
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    #Configura la recoleccion de estadisticas sobre el tamano (numero de nodos) de los individuos.
    stats_size = tools.Statistics(len)
    #Combina ambas estadisticas en un solo objeto para mostrarlas juntas.
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    #Registra las metricas a calcular: promedio, desviacion estandar, minimo y maximo.
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    #AQUÍ ESTÁ EL CAMBIO: Se usa algorithms.eaSimple en lugar de gp.eaSimple.
    #Ejecuta el algoritmo evolutivo con la poblacion, toolbox, prob. de cruce, prob. de mutacion y # de generaciones.
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)

    #Imprime una cabecera para el resultado final.
    print("\n--- Mejor Individuo ---")
    #Imprime el mejor individuo encontrado, que esta guardado en el Hall of Fame.
    print(hof[0])

    #Devuelve la poblacion final, el registro de estadisticas y el Hall of Fame.
    return pop, log, hof

# --- Ejecución del programa ---
#Asegura que el siguiente bloque de codigo solo se ejecute si el script es el principal.
if __name__ == "__main__":
    #Llama a la funcion principal para iniciar el programa.
    main()