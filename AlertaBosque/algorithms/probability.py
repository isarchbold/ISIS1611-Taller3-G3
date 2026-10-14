"""Punto 1. Distribuciones discretas y decisiones de respuesta."""
import math


def normalize(weights: dict) -> dict:
    """Retorna un diccionario nuevo cuyas probabilidades suman uno.

    Entrada: weights = {resultado: peso}.
    Salida: {resultado: probabilidad}, con las mismas claves.

    Tips:
    - Conserve las claves y su orden; no modifique weights.
    - Rechace pesos negativos, no finitos, diccionarios vacíos y suma cero.
    """
    if not weights:
        raise ValueError("El diccionario está vacío")

    for peso in weights.values():
        if peso < 0 or not math.isfinite(peso):
            raise ValueError("Peso inválido")

    total = sum(weights.values())

    if total == 0:
        raise ValueError("La suma es cero")

    rta = {}

    for clave, peso in weights.items():
        rta[clave] = peso / total

    return rta


def expected_utility(distribution: dict, utilities: dict) -> float:
    """Calcula sum_s P(s) U(s) para una acción.

    Entradas: distribution = {resultado: probabilidad} y
    utilities = {resultado: utilidad} para una sola acción.
    Salida: un número con la utilidad esperada.

    Tips:
    - distribution ya debe estar normalizada; no normalice utilidades.
    - Ambas tablas deben contener los mismos resultados posibles.
    """

    if set(distribution) != set(utilities) or not distribution:
        raise ValueError("Probabilidades y utilidades deben tener las mismas claves")
    if any(not math.isfinite(p) or p < 0 for p in distribution.values()):
        raise ValueError("Distribución inválida")
    if not math.isclose(sum(distribution.values()), 1, abs_tol=1e-9):
        raise ValueError("La distribución debe sumar uno")
    if any(not math.isfinite(u) for u in utilities.values()):
        raise ValueError("Utilidades no finitas")

    rta = 0

    for estado, probabilidad in distribution.items():
        rta += probabilidad * utilities[estado]

    return rta


def best_action(distribution: dict, actions: dict) -> tuple[str, dict]:
    """Retorna (nombre de acción, utilidades esperadas por acción).

    Entradas: distribution = {resultado: probabilidad} y
    actions = {acción: {resultado: utilidad}}.
    Salida: (acción elegida, {acción: utilidad esperada}).

    Tips:
    - actions asocia cada acción con su tabla de utilidades.
    - Reutilice expected_utility y conserve la primera acción en los empates.
    """
    # Validación de las acciones; no modificar.
    if not actions:
        raise ValueError("Se requiere al menos una acción")
    rta = {}
    mejor_accion = None
    mejor_utilidad = float("-inf")

    for accion, utilidades in actions.items():
        utilidad = expected_utility(distribution, utilidades)
        rta[accion] = utilidad

        if utilidad > mejor_utilidad:
            mejor_utilidad = utilidad
            mejor_accion = accion

    return mejor_accion, rta