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

    def consultar_saldo(self):
        print(f"Saldo atual: R$ {self._saldo}")


class ContaPoupanca(ContaBancaria):
    def __init__(self, numero, titular, saldo_inicial, taxa_juros):
        super().__init__(numero, titular, saldo_inicial)
        self.taxa_juros = taxa_juros

    def calcular_juros(self):
        juros = (self._saldo * self.taxa_juros) / 100
        self._saldo += juros
        print(
            f"Juros de R$ {juros} aplicados. Novo saldo: R$ {self._saldo:.2f}"
        )


class ContaCorrente(ContaBancaria):
    def __init__(self, numero, titular, saldo_inicial, limite_cheque_especial):
        super().__init__(numero, titular, saldo_inicial)
        self.limite_total = limite_cheque_especial
        self._limite_disponivel = limite_cheque_especial

    def saque(self, valor_desejado):
        if self._saldo >= valor_desejado:
            super().saque(valor_desejado)
            print(f"Saque de R$ {valor_desejado} realizado com sucesso.")
        elif (self._saldo + self._limite_disponivel) >= valor_desejado:
            valor_retirado_limite = valor_desejado - self._saldo
            self._saldo = 0
            self._limite_disponivel -= valor_retirado_limite
            print(
                f"Saque de R$ {valor_desejado} realizado com sucesso utilizando o cheque especial."
            )
            print(f"Limite disponível restante: R$ {self._limite_disponivel}")
        else:
            print("Saldo insuficiente para o saque.")

    def deposito(self, valor_depositado):
        if valor_depositado > 0:
            limite_utilizado = self.limite_total - self._limite_disponivel
            if limite_utilizado > 0:
                valor_a_repor = min(valor_depositado, limite_utilizado)
                self._limite_disponivel += valor_a_repor
                valor_restante = valor_depositado - valor_a_repor
                print(
                    f"R$ {valor_a_repor} utilizado para restaurar o limite do cheque especial."
                )

                if valor_restante > 0:
                    self._saldo += valor_restante
                    print(
                        f"Depósito de R$ {valor_restante} realizado com sucesso."
                    )
            else:
                self._saldo += valor_depositado
                print(
                    f"Depósito de R$ {valor_depositado} realizado com sucesso."
                )
        else:
            print("Valor de depósito inválido.")

    def consultar_saldo(self):
        super().consultar_saldo()
        print(f"Limite disponível: R$ {self._limite_disponivel}")


if __name__ == "__main__":
    cb = ContaCorrente(1, "vanessa", 1000)
    cb.consultar_saldo()
