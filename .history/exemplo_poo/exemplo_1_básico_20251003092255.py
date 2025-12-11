class Documento:
    """Classe que representa um documento com título e conteúdo."""

    def __init__(self, texto):
        """Inicializa o documento com o texto fornecido."""
        self.texto = texto
        self.tokens = self._tokenizar()

    def procurar_letra(self, letra):
        """procura por uma letra no texto e retorna a quantidade de ocorrências."""
        return self.texto.lower().count(letra.lower())

    def _tokenizar(self):
        """Tokeniza o texto em palavras."""
        return self.texto.split()


if __name__ == "__main__":
    objeto_documento = Documento("Este é um exemplo simples de documento.")
