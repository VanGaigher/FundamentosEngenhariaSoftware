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

    @staticmethod
    def validar_extensao(nome_arquivo):
        """
        Método estático: não precisa de uma instância (self) nem da classe (cls).
        É uma função utilitária relacionada à classe.
        """
        return nome_arquivo.lower().endswith(".txt")

    # cls será Documento
    @classmethod
    def from_file(cls, caminho_arquivo):
        """
        Método de classe: recebe a classe (cls) como primeiro argumento.
        Usado como um construtor alternativo.
        """
        if not cls.validar_extensao(caminho_arquivo):
            raise ValueError("Formato de arquivo inválido, esperado .txt")

        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            texto = f.read()

        # 'cls' aqui é a própria classe Documento.
        # Retorna uma nova instância da classe.
        return cls(texto)


if __name__ == "__main__":
    # Não preciso fazer documento = Document(...)
    print(Documento.validar_extensao("meu_doc.mkv"))

    # Criando um objeto a partir de um arquivo
    documento_por_arquivo = Documento.from_file(
        r"C:\Users\renne\Desktop\aulas\fundamentos-engenharia-software\exemplos_poo\meu_arquivo.txt"
    )

    print(documento_por_arquivo.texto)
    print(documento_por_arquivo.tokens)
