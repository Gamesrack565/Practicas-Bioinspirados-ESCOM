import numpy as np
import sys
import multiprocessing as mp  # multiprocessing

# --- 1. Definición de la Función Objetivo ---
def objective_function(particle):
    x = particle[0]
    y = particle[1]
    return x ** 2 + y ** 2 + (25 * (np.sin(x) + np.sin(y)))


def run_pso():
    # --- 2. Parámetros de la Práctica ---
    N_PARTICLES = 20
    N_ITERATIONS = 50
    DIMENSIONS = 2

    MIN_BOUND = -5.0
    MAX_BOUND = 5.0

    W = 0.8   # Inercia
    C1 = 0.7  # Aprendizaje local
    C2 = 1.2  # Aprendizaje global

    # --- 3. Inicialización del Enjambre ---
    particles_pos = np.random.uniform(MIN_BOUND, MAX_BOUND, (N_PARTICLES, DIMENSIONS))
    particles_vel = np.zeros((N_PARTICLES, DIMENSIONS))
    particles_pbest_pos = np.copy(particles_pos)
    particles_pbest_val = np.full(N_PARTICLES, sys.float_info.max)

    for i in range(N_PARTICLES):
        particles_pbest_val[i] = objective_function(particles_pbest_pos[i])

    gbest_index = np.argmin(particles_pbest_val)
    gbest_pos = np.copy(particles_pbest_pos[gbest_index])
    gbest_val = particles_pbest_val[gbest_index]

    print(f"Iniciando PSO Global (gbest) con {N_PARTICLES} partículas y {N_ITERATIONS} iteraciones.")
    print(f"Función a minimizar: x^2 + y^2 + [25 * (sen(x) + sen(y))]")
    print(f"Parámetros: W(a)={W}, C1(b1)={C1}, C2(b2)={C2}")

    # --- Lista para guardar los detalles de cada iteración ---
    historial = []

    # --- Crear un pool una sola vez ---
    with mp.Pool() as pool:
        for i in range(N_ITERATIONS):
            for j in range(N_PARTICLES):
                r1 = np.random.rand(DIMENSIONS)
                r2 = np.random.rand(DIMENSIONS)

                cognitive_component = C1 * r1 * (particles_pbest_pos[j] - particles_pos[j])
                social_component = C2 * r2 * (gbest_pos - particles_pos[j])

                new_velocity = (W * particles_vel[j]) + cognitive_component + social_component
                particles_vel[j] = new_velocity

                new_position = particles_pos[j] + new_velocity
                new_position = np.clip(new_position, MIN_BOUND, MAX_BOUND)
                particles_pos[j] = new_position

            # Evaluar TODAS las partículas en paralelo
            new_values = pool.map(objective_function, particles_pos)

            # Actualizar pbest
            for j in range(N_PARTICLES):
                if new_values[j] < particles_pbest_val[j]:
                    particles_pbest_val[j] = new_values[j]
                    particles_pbest_pos[j] = particles_pos[j]

            # Actualizar gbest
            current_best_index = np.argmin(particles_pbest_val)
            if particles_pbest_val[current_best_index] < gbest_val:
                gbest_val = particles_pbest_val[current_best_index]
                gbest_pos = particles_pbest_pos[current_best_index]

            # Guardar datos de esta iteración
            historial.append({
                "iter": i + 1,
                "positions": np.copy(particles_pos),
                "velocities": np.copy(particles_vel),
                "pbest": np.copy(particles_pbest_pos),
                "gbest_pos": np.copy(gbest_pos),
                "gbest_val": gbest_val
            })

            # Mostrar resumen corto
            print(f"\n==================== ITERACIÓN {i + 1} / {N_ITERATIONS} ====================")
            print(f"Mejor posición global (gbest): {gbest_pos}  |  Valor mínimo: {gbest_val:.6f}")

    # --- 5. Resultado Final ---
    print("\n==================== RESULTADO FINAL ====================")
    print(f"Optimización completada después de {N_ITERATIONS} iteraciones.")
    print(f"La mejor posición (gbest) encontrada es: {gbest_pos}")
    print(f"El valor mínimo (gbest_val) encontrado es: {gbest_val}")

    # --- 6. Mostrar detalles opcionales ---
    opcion = input("\n¿Quieres ver los detalles de las iteraciones? (S/N): ").strip().lower()
    if opcion == 's':
        sub_opcion = input("¿Deseas ver (A) todas o (E) una iteración específica?: ").strip().lower()

        if sub_opcion == 'a':
            # Mostrar todas las iteraciones
            for data in historial:
                mostrar_detalles_iteracion(data, N_ITERATIONS)
        elif sub_opcion == 'e':
            try:
                num = int(input(f"Ingrese el número de iteración (1-{N_ITERATIONS}): "))
                if 1 <= num <= N_ITERATIONS:
                    mostrar_detalles_iteracion(historial[num - 1], N_ITERATIONS)
                else:
                    print("⚠️ Iteración fuera de rango.")
            except ValueError:
                print("⚠️ Entrada no válida. Debes ingresar un número.")
        else:
            print("Opción no válida. Finalizando.")


# --- Función auxiliar para imprimir una iteración completa ---
def mostrar_detalles_iteracion(data, total_iters):
    print(f"\n=================== ITERACIÓN {data['iter']} / {total_iters} ====================")
    print("\n--- Posición de las partículas (x, y) ---")
    print(data['positions'])
    print("\n--- Velocidad de las partículas (vx, vy) ---")
    print(data['velocities'])
    print("\n--- Mejores Posiciones Personales (pbest) ---")
    print(data['pbest'])
    print("\n--- Mejor Posición Global (gbest) ---")
    print(f"Posición (x, y): {data['gbest_pos']}")
    print(f"Valor (mínimo): {data['gbest_val']:.6f}")


# --- Punto de entrada ---
if __name__ == "__main__":
    mp.freeze_support()
    run_pso()
