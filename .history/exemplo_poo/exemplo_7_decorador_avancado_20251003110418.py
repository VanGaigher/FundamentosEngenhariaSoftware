def saudar():
    print("Olá, mundo!")


def meu_decorador(funcao_original):
    # O 'wrapper' é a função "embrulhada" que será retornada
    def wrapper():
        print("Algo acontece ANTES da função original ser chamada.")
        funcao_original()
        print("Algo acontece DEPOIS da função original ser chamada.")

    return wrapper


# Agora, vamos "decorar" a função manualmente
saudar_decorada = meu_decorador(saudar)

# def saudar_decorada():
#     print("Algo acontece ANTES da função original ser chamada.")
#     print("Olá, mundo!")
#     print("Algo acontece DEPOIS da função original ser chamada.")

#################################################

import time
import functools


def timer(func):
    """Um decorador que imprime o tempo de execução de uma função."""

    # Boa prática! Preserva os metadados da função original.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()

        # Executa a função original, passando quaisquer argumentos
        resultado = func(*args, **kwargs)

        fim = time.perf_counter()
        tempo_total = fim - inicio
        print(
            f"Função '{func.__name__}' levou {tempo_total:.4f} segundos para executar."
        )

        return resultado

    return wrapper


@timer
def processo_demorado(segundos):
    """Uma função que simula um processo demorado."""
    print(f"Processando por {segundos} segundo(s)...")
    time.sleep(segundos)
    return "Processo finalizado!"


@timer
def treino_de_modelo():
    """Uma função que simula um processo demorado."""
    print(f"Treinando modelo...")
    return "Processo finalizado!"


@timer
def treino_de_modelo():
    """Uma função que simula um processo demorado."""
    print(f"Treinando modelo...")
    return "Processo finalizado!"


if __name__ == "__main__":
    # saudar_decorada()

    resultado_final = processo_demorado(2)
    print(f"Resultado: {resultado_final}")

    resultado_treino = treino_de_modelo()
