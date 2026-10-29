"""Punto 2. Inferencia exacta por enumeración."""
from algorithms.probability import normalize

#Esqueleto sacado de  Notas_de_Clase_v4 (Cap 6.4, muestra un pseudocodigo general, más no para el caso recursivo)
def enumeration_ask(network, query: str, evidence: dict) -> dict:
    """Calcula P(query | evidence) sobre la conjunta factorizada.

    Entradas: network es la red; query es el nombre de la
    variable consultada; evidence es {nombre de variable: 0 o 1}.
    Salida: {0: probabilidad, 1: probabilidad}.

    Tips:
    - Recorra network.variables en orden topológico.
    - Para una variable fijada multiplique su probabilidad; para una oculta
      sume las dos alternativas 0 y 1. Normalice al terminar.
    - Use copias del evento y no modifique evidence.
    - network.probability(variable, valor, evento) consulta la CPT;
      el evento debe contener los valores de todos los padres.
    - Puede usar una función auxiliar recursiva. Al terminar el recorrido,
      el producto sin factores restantes vale 1.
    - Evidencia imposible: ValueError. Consulta observada: distribución puntual
      solo si la evidencia es posible.
    """
    # Valida nombres de variables y dominio de la evidencia.
    network.validate_query(query, evidence)
    pesos = {}
    for valor in (0, 1):
        if query in evidence:
            if evidence[query] != valor:
                pesos[valor] = 0.0
                continue
        extended_evidence = evidence.copy()
        extended_evidence[query] = valor
        pesos[valor] = enumerate_all(network.variables, extended_evidence, network)
    return normalize(pesos)

def enumerate_all(variables, evidence, network):
    """Calcula la probabilidad de un evento dado."""
    if not variables:
        return 1.0
    variable = variables[0]
    resto = variables[1:]
    if variable in evidence:
        return network.probability(variable, evidence[variable], evidence) * enumerate_all(resto, evidence, network)
    else:
        total = 0.0
        for valor in (0, 1):
            extended_evidence = evidence.copy()
            extended_evidence[variable] = valor
            total += network.probability(variable, valor, evidence) * enumerate_all(resto, extended_evidence, network)
        return total
        
    
    
    # FIN DEL BLOQUE DEL ESTUDIANTE
