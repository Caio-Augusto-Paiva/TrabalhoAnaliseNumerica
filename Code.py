import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def FatoracaoLU(A):
    # Decompoe a matriz quadrada em L e U. O for varre as colunas fazendo a eliminacao de Gauss, 
    # e os multiplicadores vao pra matriz L. O pivotamento parcial garante a estabilidade numerica.
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A deve ser quadrada.")

    L = np.eye(n)
    U = A.copy()
    Perm = list(range(n))

    for k in range(n - 1):
        PivoMax = abs(U[k, k])
        p = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > PivoMax:
                PivoMax = abs(U[i, k])
                p = i

        if PivoMax == 0.0:
            raise ValueError("Matriz singular.")

        # Faz um novo pivotamento se achar um pivo maior na coluna
        if p != k:
            U[[k, p], :] = U[[p, k], :]
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]
            Perm[k], Perm[p] = Perm[p], Perm[k]

        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] = U[k + 1:, k:] - np.outer(L[k + 1:, k], U[k, k:])

    return L, U, Perm


def ResolveLU(L, U, b, Perm=None):
    # Resolve o sistema linear em duas fases: substituicao progressiva em L e retroativa em U.
    # O vetor Perm aplica as trocas de linha durante o pivotamento.
    b = np.array(b, dtype=float)
    n = L.shape[0]
    if L.shape != (n, n) or U.shape != (n, n) or b.shape[0] != n:
        raise ValueError("Dimensoes incompativeis.")

    if Perm is not None:
        b = b[Perm]

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Matriz singular.")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def Kcond(T: float) -> float:
    return float(np.exp(T))


def FonteF(x: float, y: float) -> float:
    Sx = np.sin(np.pi * x)
    Sy = np.sin(np.pi * y)
    Cx = np.cos(np.pi * x)
    Cy = np.cos(np.pi * y)
    T = Sx * Sy
    return float(np.exp(T) * (np.pi**2) * (Cx**2 * Sy**2 + Sx**2 * Cy**2 - 2 * Sx * Sy))


def CalcularF(U: np.ndarray, N: int) -> np.ndarray:
    # Calcula o vetor de residuos F(U) aproximando as derivadas por diferencas finitas centradas.
    # As condutividades Kcond sao calculadas nos pontos medios dos nos da malha para segurar a ordem de precisao.
    U = np.array(U, dtype=float).reshape((N, N))
    h = 1.0 / (N + 1)
    T = np.zeros((N + 2, N + 2))
    T[1: N + 1, 1: N + 1] = U

    F = np.zeros((N, N))

    for i in range(1, N + 1):
        x = i * h
        for j in range(1, N + 1):
            y = j * h
            Tij = T[i, j]
            Tip = T[i + 1, j]
            Tim = T[i - 1, j]
            Tjp = T[i, j + 1]
            Tjm = T[i, j - 1]

            Kij = Kcond(Tij)
            Kip = Kcond(Tip)
            Kim = Kcond(Tim)
            Kjp = Kcond(Tjp)
            Kjm = Kcond(Tjm)

            Kip12 = 0.5 * (Kip + Kij)
            Kim12 = 0.5 * (Kim + Kij)
            Kjp12 = 0.5 * (Kjp + Kij)
            Kjm12 = 0.5 * (Kjm + Kij)

            Termo = (
                Kip12 * (Tip - Tij)
                - Kim12 * (Tij - Tim)
                + Kjp12 * (Tjp - Tij)
                - Kjm12 * (Tij - Tjm)
            )

            F[i - 1, j - 1] = (Termo / (h**2)) - FonteF(x, y)

    return F.reshape(N * N)


def CalcularJacobiana(U: np.ndarray, N: int) -> np.ndarray:
    # Monta a matriz Jacobiana avaliando as derivadas de F em relacao as temperaturas.
    # Por ser um problema 2D, so os vizinhos adjacentes entram no preenchimento (matriz esparsa de 5 diagonais).
    # Como o problema e nao linear, a regra da cadeia entra no calculo dos termos de derivada.
    U = np.array(U, dtype=float).reshape((N, N))
    h = 1.0 / (N + 1)

    T = np.zeros((N + 2, N + 2))
    T[1: N + 1, 1: N + 1] = U

    J = np.zeros((N * N, N * N))
    InvH2 = 1.0 / (h * h)

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            Tij = T[i, j]
            Tip = T[i + 1, j]
            Tim = T[i - 1, j]
            Tjp = T[i, j + 1]
            Tjm = T[i, j - 1]

            Kij = Kcond(Tij)
            Kip = Kcond(Tip)
            Kim = Kcond(Tim)
            Kjp = Kcond(Tjp)
            Kjm = Kcond(Tjm)

            Kip12 = 0.5 * (Kip + Kij)
            Kim12 = 0.5 * (Kim + Kij)
            Kjp12 = 0.5 * (Kjp + Kij)
            Kjm12 = 0.5 * (Kjm + Kij)

            Dij = (
                0.5 * Kij * (Tip - Tij) - Kip12
                - 0.5 * Kij * (Tij - Tim) - Kim12
                + 0.5 * Kij * (Tjp - Tij) - Kjp12
                - 0.5 * Kij * (Tij - Tjm) - Kjm12
            )
            Dip = 0.5 * Kip * (Tip - Tij) + Kip12
            Dim = -0.5 * Kim * (Tij - Tim) + Kim12
            Djp = 0.5 * Kjp * (Tjp - Tij) + Kjp12
            Djm = -0.5 * Kjm * (Tij - Tjm) + Kjm12

            # Faz a passagem dos indices 2D da malha pra estrutura 1D da matriz
            LinhaK = (i - 1) * N + (j - 1)
            J[LinhaK, LinhaK] = Dij * InvH2

            if i + 1 <= N:
                ColunaK = i * N + (j - 1)
                J[LinhaK, ColunaK] = Dip * InvH2
            if i - 1 >= 1:
                ColunaK = (i - 2) * N + (j - 1)
                J[LinhaK, ColunaK] = Dim * InvH2
            if j + 1 <= N:
                ColunaK = (i - 1) * N + j
                J[LinhaK, ColunaK] = Djp * InvH2
            if j - 1 >= 1:
                ColunaK = (i - 1) * N + (j - 2)
                J[LinhaK, ColunaK] = Djm * InvH2

    return J


def NormaInfinito(v: np.ndarray) -> float:
    v = np.array(v, dtype=float)
    return float(np.max(np.abs(v)))


def InferirN(U: np.ndarray) -> int:
    m = int(U.size)
    n = int(np.sqrt(m))
    if n * n != m:
        raise ValueError("U deve ter tamanho N^2.")
    return n


def BiCGSTAB(
    A: np.ndarray, b: np.ndarray, Tol: float, MaxIterBCG: int
) -> tuple[np.ndarray, int]:
    # Metodo iterativo de Krylov (Gradiente Biconjugado Estabilizado) pra sistemas assimetricos.
    # O laco atualiza os vetores de direcao p e de residuo r pra tentar minimizar o erro ate a tolerancia limite.
    # É otimo porque a gente nao precisa da matriz na memoria toda estruturada, economizando muito espaco.
    x = np.zeros_like(b, dtype=float)
    r = b - np.dot(A, x)
    R0 = r.copy()
    Rho = 1.0
    Alfa = 1.0
    Omega = 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)

    NormaB = np.sqrt(float(np.dot(b, b)))
    if NormaB == 0.0:
        NormaB = 1.0

    for k in range(MaxIterBCG):
        RhoNovo = float(np.dot(R0, r))
        if abs(RhoNovo) < 1e-40:
            break

        Beta = (RhoNovo / Rho) * (Alfa / Omega)
        p = r + Beta * (p - Omega * v)

        v = np.dot(A, p)
        Denom = float(np.dot(R0, v))
        if abs(Denom) < 1e-40:
            break
        Alfa = RhoNovo / Denom

        s = r - Alfa * v
        NormaS = np.sqrt(float(np.dot(s, s)))
        if NormaS / NormaB < Tol:
            x = x + Alfa * p
            return x, k + 1

        t = np.dot(A, s)
        DenomOmega = float(np.dot(t, t))
        if abs(DenomOmega) < 1e-40:
            x = x + Alfa * p
            return x, k + 1
        Omega = float(np.dot(t, s)) / DenomOmega

        x = x + Alfa * p + Omega * s
        r = s - Omega * t

        NormaR = np.sqrt(float(np.dot(r, r)))
        if NormaR / NormaB < Tol:
            return x, k + 1

        if abs(Omega) < 1e-40:
            break

        Rho = RhoNovo

    return x, MaxIterBCG


def NewtonLU(U0: np.ndarray, Tol: float, MaxIter: int) -> tuple[np.ndarray, int, list]:
    # O Newton-Raphson tradicional avaliando a Jacobiana completa no laco iterativo.
    # Resolve J * DeltaU = -F  pela fatoracao LU. Otima convergencia se o chute for bom, 
    # porem recriar a Jacobiana e fatorar do zero em cada volta e bem pesado.
    U = np.array(U0, dtype=float).copy()
    N = InferirN(U)
    HistoricoRes = []

    for k in range(MaxIter):
        F = CalcularF(U, N)
        ResNorma = NormaInfinito(F)
        HistoricoRes.append(ResNorma)
        if ResNorma < Tol:
            return U, k, HistoricoRes

        J = CalcularJacobiana(U, N)
        L, ULU, Perm = FatoracaoLU(J)

        DeltaU = ResolveLU(L, ULU, -F, Perm)
        U = U + DeltaU

    return U, MaxIter, HistoricoRes


def NewtonBCG(U0: np.ndarray, Tol: float, MaxIter: int) -> tuple[np.ndarray, int, list]:
    # O laco continua montando J, mas acha DeltaU de maneira aproximada e iterativa.
    U = np.array(U0, dtype=float).copy()
    N = InferirN(U)
    MaxIterBCG = max(50, 2 * U.size)
    HistoricoRes = []

    for k in range(MaxIter):
        F = CalcularF(U, N)
        ResNorma = NormaInfinito(F)
        HistoricoRes.append(ResNorma)
        if ResNorma < Tol:
            return U, k, HistoricoRes

        J = CalcularJacobiana(U, N)

        DeltaU, _ = BiCGSTAB(J, -F, Tol * 0.1, MaxIterBCG)
        U = U + DeltaU

    return U, MaxIter, HistoricoRes


def Broyden(U0: np.ndarray, Tol: float, MaxIter: int) -> tuple[np.ndarray, int, list]:
    # Metodo de Broyden. Calcula J so na primeira iteracao.
    # Nas seguintes iteracoes do loop, a aproximacao da Jacobiana e atualizada usando a equacao secante com a diferenca do residuo.
    # A velocidade da volta e maior por nao recalcular J inteira, embora precisa dar mais passos pra convergir.
    U = np.array(U0, dtype=float).copy()
    N = InferirN(U)
    HistoricoRes = []

    F = CalcularF(U, N)
    ResNorma = NormaInfinito(F)
    HistoricoRes.append(ResNorma)
    if ResNorma < Tol:
        return U, 0, HistoricoRes

    J = CalcularJacobiana(U, N)

    for k in range(MaxIter):
        if NormaInfinito(F) < Tol:
            return U, k, HistoricoRes

        L, ULU, Perm = FatoracaoLU(J)
        DeltaU = ResolveLU(L, ULU, -F, Perm)

        UNovo = U + DeltaU
        FNovo = CalcularF(UNovo, N)

        # Atualizacao do posto 1 da aproximacao da matriz baseada nas informacoes de secante 
        s = UNovo - U
        y = FNovo - F
        Denom = float(np.dot(s, s))
        if Denom != 0.0:
            J = J + np.outer((y - np.dot(J, s)), s) / Denom

        U = UNovo
        F = FNovo
        HistoricoRes.append(NormaInfinito(F))

    return U, MaxIter, HistoricoRes


def SolucaoExata(N: int) -> np.ndarray:
    h = 1.0 / (N + 1)
    T = np.zeros((N, N))
    for i in range(1, N + 1):
        x = i * h
        for j in range(1, N + 1):
            y = j * h
            T[i - 1, j - 1] = np.sin(np.pi * x) * np.sin(np.pi * y)
    return T.reshape(N * N)


def FloatParaExcel(Valor):
    if Valor is None:
        return "-"
    return f"{Valor}".replace(".", ",")


def SalvarCSV(Resultados: list, ErrosLU: list, Hs: list, Ns: list) -> None:
    with open("resultados.csv", "w", newline="", encoding="utf-8-sig") as Arq:
        Escritor = csv.writer(Arq, delimiter=";")
        Escritor.writerow(["N", "Metodo", "Iteracoes", "Tempo(s)", "Erro Maximo"])
        for N, Nome, It, Tempo, Erro in Resultados:
            ItStr = str(It) if It is not None else "-"
            TempoStr = FloatParaExcel(round(Tempo, 6)) if Tempo is not None else "-"
            ErroStr = FloatParaExcel(Erro) if Erro is not None else "-"
            Escritor.writerow([N, Nome, ItStr, TempoStr, ErroStr])

        Escritor.writerow([])
        Escritor.writerow(["Ordem de convergencia (Newton-LU)"])
        Escritor.writerow(["De", "Para", "p"])
        for i in range(len(ErrosLU) - 1):
            if ErrosLU[i] is None or ErrosLU[i + 1] is None:
                continue
            p = np.log(ErrosLU[i] / ErrosLU[i + 1]) / np.log(Hs[i] / Hs[i + 1])
            Escritor.writerow([
                f"N={Ns[i]}",
                f"N={Ns[i+1]}",
                FloatParaExcel(round(p, 4))
            ])


def PlotarCorte1D(UNum: np.ndarray, UExato: np.ndarray, N: int) -> None:
    UNum2D = np.array(UNum, dtype=float).reshape((N, N))
    UExato2D = np.array(UExato, dtype=float).reshape((N, N))

    TNum = np.zeros((N + 2, N + 2))
    TEx = np.zeros((N + 2, N + 2))
    TNum[1: N + 1, 1: N + 1] = UNum2D
    TEx[1: N + 1, 1: N + 1] = UExato2D

    x = np.linspace(0.0, 1.0, N + 2)
    IndiceCorte = (N + 2) // 2

    Fig, Ax = plt.subplots(1, 2, figsize=(14, 5))

    Ax[0].plot(x, TNum[:, IndiceCorte], "b-o", markersize=4, linewidth=2, label="Numerica")
    Ax[0].plot(x, TEx[:, IndiceCorte], "r--", linewidth=2, label="Exata")
    Ax[0].set_title(f"Corte em y = 0.5 (N={N})", fontsize=14)
    Ax[0].set_xlabel("x", fontsize=12)
    Ax[0].set_ylabel("T(x, 0.5)", fontsize=12)
    Ax[0].legend(fontsize=11)
    Ax[0].grid(True, linestyle="--", alpha=0.5)

    ErroCorte = np.abs(TNum[:, IndiceCorte] - TEx[:, IndiceCorte])
    Ax[1].plot(x, ErroCorte, "k-s", markersize=4, linewidth=2)
    Ax[1].set_title(f"Erro no corte y = 0.5 (N={N})", fontsize=14)
    Ax[1].set_xlabel("x", fontsize=12)
    Ax[1].set_ylabel("|T_num - T_exata|", fontsize=12)
    Ax[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("Corte1D.png", dpi=150)
    plt.show()


def PlotarSuperficie3D(UNum: np.ndarray, UExato: np.ndarray, N: int) -> None:
    UNum2D = np.array(UNum, dtype=float).reshape((N, N))
    UExato2D = np.array(UExato, dtype=float).reshape((N, N))

    TNum = np.zeros((N + 2, N + 2))
    TEx = np.zeros((N + 2, N + 2))
    TNum[1: N + 1, 1: N + 1] = UNum2D
    TEx[1: N + 1, 1: N + 1] = UExato2D

    x = np.linspace(0.0, 1.0, N + 2)
    y = np.linspace(0.0, 1.0, N + 2)
    X, Y = np.meshgrid(x, y, indexing="ij")

    Fig = plt.figure(figsize=(16, 6))

    Ax1 = Fig.add_subplot(1, 2, 1, projection="3d")
    Ax1.plot_surface(X, Y, TNum, cmap="viridis", alpha=0.9, edgecolor="none")
    Ax1.set_title(f"Solucao Numerica (N={N})", fontsize=13)
    Ax1.set_xlabel("x")
    Ax1.set_ylabel("y")
    Ax1.set_zlabel("T(x, y)")

    Erro = np.abs(TNum - TEx)
    Ax2 = Fig.add_subplot(1, 2, 2, projection="3d")
    Ax2.plot_surface(X, Y, Erro, cmap="hot", alpha=0.9, edgecolor="none")
    Ax2.set_title(f"Erro Pontual (N={N})", fontsize=13)
    Ax2.set_xlabel("x")
    Ax2.set_ylabel("y")
    Ax2.set_zlabel("|T_num - T_exata|")

    plt.tight_layout()
    plt.savefig("Superficie3D.png", dpi=150)
    plt.show()


def PlotarConvergencia(HistoricoMetodos: dict, N: int) -> None:
    Fig, Ax = plt.subplots(figsize=(9, 6))

    Cores = {"Newton-LU": "#2563eb", "Newton-BCG": "#16a34a", "Broyden": "#dc2626"}
    Marcadores = {"Newton-LU": "o", "Newton-BCG": "s", "Broyden": "^"}

    for Nome, Historico in HistoricoMetodos.items():
        Iteracoes = list(range(len(Historico)))
        Ax.semilogy(
            Iteracoes, Historico,
            marker=Marcadores.get(Nome, "o"),
            color=Cores.get(Nome, "#000000"),
            linewidth=2, markersize=6,
            label=Nome
        )

    Ax.axhline(y=1e-6, color="gray", linestyle=":", linewidth=1.5, label="Tolerancia (1e-6)")
    Ax.set_title(f"Norma do Residuo por Iteracao (N={N})", fontsize=14)
    Ax.set_xlabel("Iteracao", fontsize=12)
    Ax.set_ylabel("||F(U)||_inf", fontsize=12)
    Ax.legend(fontsize=11)
    Ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("ConvergenciaResiduos.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    Ns = [4, 8, 16, 32]
    Tol = 1e-6
    MaxIter = 100

    Resultados = []
    ErrosLU = []
    Hs = []

    NPlot = 16
    UPlotLU = None
    UExataPlot = None
    HistoricoPlot = {}

    for N in Ns:
        h = 1.0 / (N + 1)
        Hs.append(h)

        U0 = np.zeros(N * N)
        UExata = SolucaoExata(N)

        for Nome, Metodo in (
            ("Newton-LU", NewtonLU),
            ("Newton-BCG", NewtonBCG),
            ("Broyden", Broyden),
        ):
            try:
                T0 = time.time()
                UFinal, It, Historico = Metodo(U0, Tol, MaxIter)
                T1 = time.time()

                Erro = NormaInfinito(UFinal - UExata)
                Resultados.append((N, Nome, It, T1 - T0, Erro))

                if Nome == "Newton-LU":
                    ErrosLU.append(Erro)

                if N == NPlot:
                    HistoricoPlot[Nome] = Historico
                    if Nome == "Newton-LU":
                        UPlotLU = UFinal
                        UExataPlot = UExata

            except Exception as Exc:
                Resultados.append((N, Nome, None, None, None))
                print(f"Aviso: {Nome} falhou em N={N}: {Exc}")

    print("Resultados (N, Metodo, Iteracoes, Tempo(s), Erro Maximo)")
    print("-" * 68)
    for N, Nome, It, Tempo, Erro in Resultados:
        ItStr = f"{It}" if It is not None else "-"
        TempoStr = f"{Tempo:.6f}" if Tempo is not None else "-"
        ErroStr = f"{Erro:.6e}" if Erro is not None else "-"
        print(f"{N:>4}  {Nome:<10}  {ItStr:>6}  {TempoStr:>10}  {ErroStr:>12}")

    print("\nOrdem de convergencia (Newton-LU)")
    print("-" * 40)
    for i in range(len(ErrosLU) - 1):
        if ErrosLU[i] is None or ErrosLU[i + 1] is None:
            continue
        p = np.log(ErrosLU[i] / ErrosLU[i + 1]) / np.log(Hs[i] / Hs[i + 1])
        print(f"N={Ns[i]} -> N={Ns[i+1]}: p = {p:.4f}")

    SalvarCSV(Resultados, ErrosLU, Hs, Ns)
    print("\nCSV salvo em: resultados.csv")

    if UPlotLU is not None and UExataPlot is not None:
        PlotarCorte1D(UPlotLU, UExataPlot, NPlot)
        PlotarSuperficie3D(UPlotLU, UExataPlot, NPlot)

    if HistoricoPlot:
        PlotarConvergencia(HistoricoPlot, NPlot)