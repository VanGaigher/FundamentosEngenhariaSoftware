class ContaBancaria:
    def __init__(self, numero, titular, saldo_inicial):
        self.numero = numero
        self.titular = titular
        self._saldo = saldo_inicial

    def saque(self, valor_desejado):
        if self._saldo >= valor_desejado:
            self.saldo -= valor_desejado
            print(f"Saque de R$ {valor_desejado} realizado com sucesso.")
            return valor_desejado
        else:
            print("Saldo insuficiente para o saque.")

    def deposito(self, valor_depositado):
        if valor_depositado > 0:
            self._saldo += valor_depositado
            print(f"Depósito de R$ {valor_depositado} realizado com sucesso.")
        else:
            print("Valor de depósito inválido.")
