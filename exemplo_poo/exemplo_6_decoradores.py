class Documento:
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

    @property
    def contagem_palavras(self):
        """
        Este método pode ser acessado como um atributo, sem parênteses.
        Ex: meu_doc.contagem_palavras
        """
        return len(self.tokens)


if __name__ == "__main__":
    doc = Documento("Este é um texto de exemplo.")

    print(f"Contagem de palavras: {doc.contagem_palavras}")
