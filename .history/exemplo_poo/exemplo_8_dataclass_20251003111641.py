from dataclasses import dataclass
from datetime import date


@dataclass
class MetadadosDocumento:
    """
    Uma dataclass para armazenar metadados de forma simples e clara.
    O __init__, __repr__ e outros métodos são criados automaticamente.
    """

    autor: str
    data_criacao: date
    versao: int = 1


if __name__ == "__main__":
    meta = MetadadosDocumento(autor="Renata", data_criacao=date.today())
    print(meta)
