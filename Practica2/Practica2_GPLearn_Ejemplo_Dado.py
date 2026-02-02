#@Angel Abraham Higuera Pineda

#Importa la biblioteca NumPy para operaciones numericas, especialmente con arreglos.
import numpy as np
#Importa la biblioteca Matplotlib para la creacion de graficos y visualizaciones.
import matplotlib.pyplot as plt
#Importa la biblioteca Graphviz para visualizar graficos de arbol.
import graphviz

#Importa la clase principal para la regresion simbolica de la biblioteca gplearn.
from gplearn.genetic import SymbolicRegressor
#Importa un modelo de Random Forest para comparar resultados.
from sklearn.ensemble import RandomForestRegressor
#Importa un modelo de Arbol de Decision para comparar.
from sklearn.tree import DecisionTreeRegressor
#Importa una utilidad para generar numeros aleatorios de forma controlada.
from sklearn.utils.random import check_random_state

# --- 1. Generación de Datos ---
#Se crea una funcion matematica base para que el algoritmo la descubra.
#f(x0, x1) = x0^2 - x1^2 + x1 - 1

#Define la funcion real (ground truth) que el sistema intentara aproximar.
#Crea un arreglo de valores para la variable x0 desde -1 a 1.
x0 = np.arange(-1, 1, 0.1)
#Crea un arreglo de valores para la variable x1 desde -1 a 1.
x1 = np.arange(-1, 1, 0.1)
#Crea una malla de coordenadas a partir de los vectores x0 y x1 para graficar en 3D.
x0, x1 = np.meshgrid(x0, x1)
#Calcula el valor real 'y' para cada punto (x0, x1) de la malla.
y_truth = x0**2 - x1**2 + x1 - 1

#Crea un generador de numeros aleatorios con una semilla fija (0) para que los resultados sean reproducibles.
rng = check_random_state(0)

#Crea las muestras de entrenamiento.
#Genera 100 valores aleatorios uniformes y los organiza en una matriz de 50x2 para X_train.
X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
#Calcula los valores 'y' correspondientes para los datos de entrenamiento usando la funcion real.
y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] - 1

#Crea las muestras de prueba (para evaluar el modelo).
#Genera otros 100 valores aleatorios y los organiza en una matriz de 50x2 para X_test.
X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
#Calcula los valores 'y' correspondientes para los datos de prueba.
y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1


# --- 2. Creación y Entrenamiento del Modelo Genético (ESTA PARTE FALTABA) ---
#Aquí se inicializa el SymbolicRegressor con sus parametros (hiperparametros).
est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

#Se entrena el modelo de regresion simbolica con los datos de entrenamiento.
print("Entrenando el SymbolicRegressor...")
est_gp.fit(X_train, y_train)
#Imprime un mensaje para indicar que el entrenamiento ha finalizado.
print("Entrenamiento completado.")

#Imprime la formula matematica que encontro el algoritmo genetico.
print("\nFórmula encontrada por el SymbolicRegressor:")
#Accede al programa (la formula) interno del regresor y lo imprime.
print(est_gp._program)


# --- 3. Entrenamiento de otros modelos para comparar ---
#Crea una instancia de un regresor de Arbol de Decision.
est_tree = DecisionTreeRegressor()
#Entrena el arbol de decision con los mismos datos de entrenamiento.
est_tree.fit(X_train, y_train)
#Crea una instancia de un regresor de Random Forest (Bosque Aleatorio).
est_rf = RandomForestRegressor(n_estimators=10, random_state=0)
#Entrena el Random Forest con los datos de entrenamiento.
est_rf.fit(X_train, y_train)


# --- 4. Predicciones y Visualización de Resultados ---
#Se realizan predicciones con todos los modelos para poder graficarlos.
#El regresor simbolico predice los valores 'y' para toda la malla de puntos.
y_gp = est_gp.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
#Calcula el puntaje R^2 del regresor simbolico usando los datos de prueba.
score_gp = est_gp.score(X_test, y_test)
#El arbol de decision predice los valores 'y' para la malla.
y_tree = est_tree.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
#Calcula el puntaje R^2 del arbol de decision.
score_tree = est_tree.score(X_test, y_test)
#El Random Forest predice los valores 'y' para la malla.
y_rf = est_rf.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
#Calcula el puntaje R^2 del Random Forest.
score_rf = est_rf.score(X_test, y_test)

#Se grafican los resultados en 3D para comparar visualmente los modelos.
#Crea una nueva figura de Matplotlib con un tamano especifico.
fig = plt.figure(figsize=(12, 10))

#Inicia un bucle para crear un subgrafico para cada modelo y la funcion original.
for i, (y, score, title) in enumerate([(y_truth, None, "Función Original (Ground Truth)"),
                                       (y_gp, score_gp, "Symbolic Regressor"),
                                       (y_tree, score_tree, "Decision Tree Regressor"),
                                       (y_rf, score_rf, "Random Forest Regressor")]):

    #Anade un nuevo subgrafico a la figura en una cuadricula de 2x2, con proyeccion 3D.
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    #Establece los limites del eje X.
    ax.set_xlim(-1, 1)
    #Establece los limites del eje Y.
    ax.set_ylim(-1, 1)
    #Define las marcas (ticks) que se mostraran en el eje X.
    ax.set_xticks(np.arange(-1, 1.01, .5))
    #Define las marcas que se mostraran en el eje Y.
    ax.set_yticks(np.arange(-1, 1.01, .5))
    #Dibuja la superficie 3D correspondiente a la funcion 'y'.
    surf = ax.plot_surface(x0, x1, y, rstride=1, cstride=1, color='green', alpha=0.5)
    #Dibuja los puntos de entrenamiento originales como un diagrama de dispersion.
    points = ax.scatter(X_train[:, 0], X_train[:, 1], y_train)
    #Si el modelo tiene un puntaje (no es la funcion original).
    if score is not None:
        #Anade un texto al grafico mostrando el puntaje R^2 formateado.
        score_text = ax.text(-.7, 1, .2, "$R^2 =\/ %.6f$" % score, 'x', fontsize=14)
    #Establece el titulo para el subgrafico actual.
    plt.title(title)

#Muestra la figura con todos los subgraficos.
plt.show()


# --- 5. Visualización del Árbol de la Solución ---
#Esta seccion exporta el mejor programa encontrado como un arbol.
#El archivo se guardara como 'symbolic_regressor_tree.png'

#Imprime un mensaje para indicar que se esta generando la grafica del arbol.
print("\nGenerando la gráfica del árbol de la solución...")
#Exporta la estructura del programa genetico al formato de lenguaje DOT.
dot_data = est_gp._program.export_graphviz()
#Crea un objeto de origen de Graphviz a partir de los datos en formato DOT.
graph = graphviz.Source(dot_data)
#Ajuste: se guarda en la carpeta actual para evitar errores si la carpeta 'images' no existe.
#Renderiza el grafico y lo guarda como un archivo PNG, eliminando los archivos intermedios.
graph.render('symbolic_regressor_tree', format='png', cleanup=True)
#Imprime un mensaje de confirmacion con el nombre del archivo guardado.
print("Gráfica guardada como 'symbolic_regressor_tree.png'")