from abc import ABC, abstractmethod


class Documento:
    """Exemplo simples de uma classe em Python. De um documento"""

    def __init__(self, texto):
        """Inicializa a classe com um texto Inicial."""
        self.texto = texto
        self.tokens = self._tokenizar()

    def procurar_letra(self, letra):
        """Procura por uma letra no texto e retorna a quantidade de ocorrências."""
        return self.texto.lower().count(letra.lower())

    def _tokenizar(self):
        """Método privado para tokenizar o texto."""
        return self.texto.split()


class DocumentoAssinavel(ABC):  # Herda de ABC para ser uma classe abstrata
    """
    Define um "contrato" para qualquer documento que possa ser assinado.
    Não pode ser instanciada diretamente.
    """

    @abstractmethod
    def assinar(self, nome):
        """Método que todas as classes filhas devem implementar."""
        pass

    @abstractmethod
    def is_valido(self):
        """Verifica se o documento é válido."""
        pass


# A classe Contrato agora implementa a interface de DocumentoAssinavel
class Contrato(Documento, DocumentoAssinavel):
    def __init__(self, texto, partes_envolvidas):
        # super() chama o construtor da classe pai (Documento)
        super().__init__(texto)

        # Adiciona novos atributos específicos do Contrato
        self.partes = partes_envolvidas
        self.__assinado = False

    def assinar(self, nome_assinante):
        if nome_assinante in self.partes:
            self.__assinado = True
            print(f"{nome_assinante} assinou o contrato.")
        else:
            print(
                f"ERRO: {nome_assinante} não é uma parte envolvida no contrato."
            )

    def is_valido(self):
        return self.__assinado


# Tentar criar uma classe filha sem implementar os métodos abstratos gera um erro.
class Proposta(Documento, DocumentoAssinavel):
    def __init__(self, texto, partes_envolvidas, orcamento):
        super().__init__(texto)

        self.partes = partes_envolvidas
        self.orcamento = orcamento
        self.__aprovado = False

    def assinar(self, nome_assinante):
        if nome_assinante in self.partes:
            self.__aprovado = True
            print(f"{nome_assinante} aprovou o orcamento de {self.orcameno}")
        else:
            print(
                f"ERRO: {nome_assinante} não é uma parte envolvida no contrato."
            )


if __name__ == "__main__":
    contrato = Contrato("Texto do Contrato", ["Parte A", "Parte B"])

    # proposta = Proposta("Texto do Contrato", ["Parte A", "Parte B"], 5000)
