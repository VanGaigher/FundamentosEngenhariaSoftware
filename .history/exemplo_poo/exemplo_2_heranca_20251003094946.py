class Documento:
    """Exemplo simples de uma classe em Python. De um documento"""

    def __init__(self, texto):
        """Inicializa o documento com o texto fornecido."""
        self.texto = texto
        self.tokens = self._tokenizar()

    def _tokenizar(self):
        """Tokeniza o texto em palavras."""
        return self.texto.split()

    def procurar_letra(self, letra):
        """procura por uma letra no texto e retorna a quantidade de ocorrências."""
        return self.texto.lower().count(letra.lower())


class Contrato(Documento):
    """Classe especializada de Documento que representa um contrato.
    Herda texto e tokens da classe pai Documento.
    """

    def __init__(self, texto, partes_envolvidas):
        # super chama o construtor da classe pai
        super().__init__(texto)
        self.__assinado = False  # exemplo para encapsulamento
        self.partes = partes_envolvidas

    def assinar(self):
        """Marca o contyrato como assinado."""

        self.__assinado = True
        print(f"Contrato entre {', '.join(self.partes)} foi assinado.")

    def is_valido(self):
        """Verifica se o contrato é válido."""
        return self.__assinado


class Ata(Documento):
    def __init__(self, texto, numero_reuniao):
        super().__init__(texto)
        self.numero_reuniao = numero_reuniao


if __name__ == "__main__":

    partes = ["Alice", "Bob"]
    contrato = Contrato("Contrato de prestação de serviços", partes)
    contrato.assinar()
    print(f"Contrato é válido? {contrato.is_valido()}")

    ata_exemplo = Ata("Ata da reunião de equipe", 42)
    print(
        f"Ata número {ata_exemplo.numero_reuniao} tokens: {ata_exemplo.tokens}"
    )

    # Tentativa de acessar o atributo privado diretamente (não recomendado)
    # print(contrato.__assinado)  # Isso causará um erro
