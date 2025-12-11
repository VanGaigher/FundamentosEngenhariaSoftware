class Documento:
    """Classe que representa um documento com título e conteúdo."""

    def __init__(self, texto):
        """INicializa a classe com um texto inicial"""
        self.texto = texto
        self.tokens = self._tokenizar()

    def procurar_letra(self, letra):
        """procura por uma letra no texto e retorna a quantidade de ocorrências."""
        return self.texto.lower().count(letra.lower())

    def _tokenizar(self):
        """Tokeniza o texto em palavras."""
        return self.texto.split()

    def get_info(self):
        """Retorna informações básicas do documento."""
        return f"Documento com {len(self.tokens)} tokens."


class ArtigoCientifico(Documento):
    """Classe especializada de Documento que representa um artigo científico.
    Herda texto e tokens da classe pai Documento.
    """

    def __init__(self, texto, autor, revista):
        super().__init__(texto)
        self.autor = autor
        self.revista = revista

    # Redefinindo um método da classe pai (polimorfismo)
    def get_info(self):
        """Retorna informações específicas do artigo científico."""
        return f"Artigo de {self.autor} publicado na revista {self.revista}."


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

    def get_info(self):
        """Retorna informações específicas do contrato."""
        status = "assinado" if self.is_valido() else "pendente"
        return f"Contrato entre {', '.join(self.partes)} está {status}."


if __name__ == "__main__":
    contrato = Contrato("Contrato de prestação de serviços", ["Alice", "Bob"])
    contrato.assinar()
    artigo = ArtigoCientifico(
        "Estudo sobre polimorfismo em Python", "Dr. Smith", "Revista de TI"
    )

    # A mesma chamada de método (.get_info()) se comporta de maneira diferente
    # dependendo do tipo do objeto (Contrato ou ArtigoCientifico)
    documento = [contrato, artigo]
    for doc in documento:
        print(doc.get_info())
