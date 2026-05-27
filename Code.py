import numpy as np
import time
import matplotlib.pyplot as plt


def fatoracao_lu(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A deve ser quadrada.")

    L = np.eye(n)
    U = A.copy()

    # Eliminacao: U recebe as eliminacoes e L guarda os multiplicadores
    for k in range(n - 1):
        if U[k, k] == 0:
            # Pivotamento parcial simples
            p = None
            for i in range(k + 1, n):
                if U[i, k] != 0:
                    p = i
                    break
            if p is None:
                raise ValueError("Matriz singular.")
            U[[k, p], :] = U[[p, k], :]
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
            U[i, k] = 0.0

    return L, U


def resolve_lu(L, U, b):
    b = np.array(b, dtype=float)
    n = L.shape[0]
    if L.shape != (n, n) or U.shape != (n, n) or b.shape[0] != n:
        raise ValueError("Dimensoes incompativeis.")

    # Substituicao Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Substituicao Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Matriz singular.")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def k_cond(T: float) -> float:
    return float(np.exp(T))


def fonte_f(x: float, y: float) -> float:
    s_x = np.sin(np.pi * x)
    s_y = np.sin(np.pi * y)
    c_x = np.cos(np.pi * x)
    c_y = np.cos(np.pi * y)
    T = s_x * s_y
    return float(-np.exp(T) * (np.pi**2) * (c_x**2 * s_y**2 + s_x**2 * c_y**2))


def calcular_F(U: np.ndarray, N: int) -> np.ndarray:
    U = np.array(U, dtype=float).reshape((N, N))
    h = 1.0 / (N + 1)

    # Conversao vetor para matriz
    T = np.zeros((N + 2, N + 2))
    T[1: N + 1, 1: N + 1] = U

    F = np.zeros((N, N))

    for i in range(1, N + 1):
        x = i * h
        for j in range(1, N + 1):
            y = j * h

            T_ij = T[i, j]
            T_ip = T[i + 1, j]
            T_im = T[i - 1, j]
            T_jp = T[i, j + 1]
            T_jm = T[i, j - 1]

            k_ij = k_cond(T_ij)
            k_ip = k_cond(T_ip)
            k_im = k_cond(T_im)
            k_jp = k_cond(T_jp)
            k_jm = k_cond(T_jm)

            k_ip12 = 0.5 * (k_ip + k_ij)
            k_im12 = 0.5 * (k_im + k_ij)
            k_jp12 = 0.5 * (k_jp + k_ij)
            k_jm12 = 0.5 * (k_jm + k_ij)

            termo = (
                k_ip12 * (T_ip - T_ij)
                - k_im12 * (T_ij - T_im)
                + k_jp12 * (T_jp - T_ij)
                - k_jm12 * (T_ij - T_jm)
            )

            F[i - 1, j - 1] = (termo / (h**2)) - fonte_f(x, y)

    return F.reshape(N * N)


def calcular_jacobiana(U: np.ndarray, N: int) -> np.ndarray:
    U = np.array(U, dtype=float).reshape((N, N))
    h = 1.0 / (N + 1)

    # Conversao vetor para matriz
    T = np.zeros((N + 2, N + 2))
    T[1: N + 1, 1: N + 1] = U

    J = np.zeros((N * N, N * N))
    inv_h2 = 1.0 / (h * h)

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            T_ij = T[i, j]
            T_ip = T[i + 1, j]
            T_im = T[i - 1, j]
            T_jp = T[i, j + 1]
            T_jm = T[i, j - 1]

            k_ij = k_cond(T_ij)
            k_ip = k_cond(T_ip)
            k_im = k_cond(T_im)
            k_jp = k_cond(T_jp)
            k_jm = k_cond(T_jm)

            k_ip12 = 0.5 * (k_ip + k_ij)
            k_im12 = 0.5 * (k_im + k_ij)
            k_jp12 = 0.5 * (k_jp + k_ij)
            k_jm12 = 0.5 * (k_jm + k_ij)

            d_ij = (
                0.5 * k_ij * (T_ip - T_ij) - k_ip12
                - 0.5 * k_ij * (T_ij - T_im) - k_im12
                + 0.5 * k_ij * (T_jp - T_ij) - k_jp12
                - 0.5 * k_ij * (T_ij - T_jm) - k_jm12
            )
            d_ip = 0.5 * k_ip * (T_ip - T_ij) + k_ip12
            d_im = -0.5 * k_im * (T_ij - T_im) + k_im12
            d_jp = 0.5 * k_jp * (T_jp - T_ij) + k_jp12
            d_jm = -0.5 * k_jm * (T_ij - T_jm) + k_jm12

            # Mapeamento (i,j) -> k_row no vetor 1D
            k_row = (i - 1) * N + (j - 1)
            J[k_row, k_row] = d_ij * inv_h2

            # Verifica se o vizinho e interno antes de preencher a coluna
            if i + 1 <= N:
                k_col = i * N + (j - 1)
                J[k_row, k_col] = d_ip * inv_h2
            if i - 1 >= 1:
                k_col = (i - 2) * N + (j - 1)
                J[k_row, k_col] = d_im * inv_h2
            if j + 1 <= N:
                k_col = (i - 1) * N + j
                J[k_row, k_col] = d_jp * inv_h2
            if j - 1 >= 1:
                k_col = (i - 1) * N + (j - 2)
                J[k_row, k_col] = d_jm * inv_h2

    return J


def norma_inf(v: np.ndarray) -> float:
    v = np.array(v, dtype=float)
    return float(np.max(np.abs(v)))


def _inferir_n(U: np.ndarray) -> int:
    m = int(U.size)
    n = int(np.sqrt(m))
    if n * n != m:
        raise ValueError("U deve ter tamanho N^2.")
    return n


def gradiente_conjugado_manual(
    A: np.ndarray, b: np.ndarray, tol: float, max_iter_cg: int
) -> tuple[np.ndarray, int]:
    x = np.zeros_like(b, dtype=float)
    r = b - np.dot(A, x)
    p = r.copy()
    rs_old = float(np.dot(r, r))
    best_x = x.copy()
    best_res = np.sqrt(rs_old)
    prev_res = best_res
    warned = False

    if best_res < tol:
        return x, 0

    for k in range(max_iter_cg):
        Ap = np.dot(A, p)
        denom = float(np.dot(p, Ap))
        if denom == 0.0:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = float(np.dot(r, r))
        res_norm = np.sqrt(rs_new)

        if res_norm < best_res:
            best_res = res_norm
            best_x = x.copy()

        if res_norm > prev_res and not warned:
            print("Aviso: residuo do CG aumentou; retornando melhor aproximacao.")
            warned = True

        if res_norm < tol:
            return x, k + 1

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
        prev_res = res_norm

    return best_x, max_iter_cg


def newton_lu(U0: np.ndarray, tol: float, max_iter: int) -> tuple[np.ndarray, int]:
    U = np.array(U0, dtype=float).copy()
    N = _inferir_n(U)

    for k in range(max_iter):
        F = calcular_F(U, N)
        if norma_inf(F) < tol:
            return U, k

        J = calcular_jacobiana(U, N)
        L, U_lu = fatoracao_lu(J)

        # resolve J(U) dU = -F(U)
        dU = resolve_lu(L, U_lu, -F)
        U = U + dU

    return U, max_iter


def newton_cg(U0: np.ndarray, tol: float, max_iter: int) -> tuple[np.ndarray, int]:
    U = np.array(U0, dtype=float).copy()
    N = _inferir_n(U)
    max_iter_cg = max(20, U.size)

    for k in range(max_iter):
        F = calcular_F(U, N)
        if norma_inf(F) < tol:
            return U, k

        J = calcular_jacobiana(U, N)

        # Passo de Newton J(U) dU = -F(U)
        dU, _ = gradiente_conjugado_manual(J, -F, tol, max_iter_cg)
        U = U + dU

    return U, max_iter


def broyden(U0: np.ndarray, tol: float, max_iter: int) -> tuple[np.ndarray, int]:
    U = np.array(U0, dtype=float).copy()
    N = _inferir_n(U)

    F = calcular_F(U, N)
    if norma_inf(F) < tol:
        return U, 0

    J = calcular_jacobiana(U, N)
    L, U_lu = fatoracao_lu(J)

    #  H = J^-1 usando J X = I
    n = J.shape[0]
    H = np.zeros((n, n))
    I = np.eye(n)
    print(
        f"  [Broyden] Calculando inversa exata para N={N} ({n}x{n}). Isso pode demorar...")
    for col in range(n):
        H[:, col] = resolve_lu(L, U_lu, I[:, col])

    for k in range(max_iter):
        if norma_inf(F) < tol:
            return U, k

        # dU = -H F
        dU = -np.dot(H, F)
        U_new = U + dU
        F_new = calcular_F(U_new, N)

        s = U_new - U
        y = F_new - F
        Hy = np.dot(H, y)
        denom = float(np.dot(s, Hy))
        if denom != 0.0:
            H = H + np.outer((s - Hy), np.dot(s, H)) / denom

        U = U_new
        F = F_new

    return U, max_iter


def _solucao_exata(N: int) -> np.ndarray:
    h = 1.0 / (N + 1)
    T = np.zeros((N, N))
    for i in range(1, N + 1):
        x = i * h
        for j in range(1, N + 1):
            y = j * h
            T[i - 1, j - 1] = np.sin(np.pi * x) * np.sin(np.pi * y)
    return T.reshape(N * N)


def plotar_resultados(U_num: np.ndarray, U_exato: np.ndarray, N: int) -> None:
    U_num = np.array(U_num, dtype=float).reshape((N, N))
    U_exato = np.array(U_exato, dtype=float).reshape((N, N))

    # Padding com zeros para representar a fronteira T=0
    T_num = np.zeros((N + 2, N + 2))
    T_ex = np.zeros((N + 2, N + 2))
    T_num[1: N + 1, 1: N + 1] = U_num
    T_ex[1: N + 1, 1: N + 1] = U_exato

    h = 1.0 / (N + 1)
    x = np.linspace(0.0, 1.0, N + 2)
    y = np.linspace(0.0, 1.0, N + 2)
    X, Y = np.meshgrid(x, y, indexing="ij")

    erro = np.abs(T_num - T_ex)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    c0 = ax[0].contourf(X, Y, T_num, levels=20)
    ax[0].set_title("Solucao Numerica")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    fig.colorbar(c0, ax=ax[0])

    c1 = ax[1].contourf(X, Y, erro, levels=20)
    ax[1].set_title("Erro Absoluto")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    fig.colorbar(c1, ax=ax[1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Ns = [4, 8, 16, 32]
    tol = 1e-6
    max_iter = 100

    resultados = []
    erros_lu = []
    hs = []

    U_plot = None
    U_exata_plot = None
    N_plot = None

    for N in Ns:
        h = 1.0 / (N + 1)
        hs.append(h)

        U0 = np.zeros(N * N)
        U_exata = _solucao_exata(N)

        for nome, metodo in (
            ("Newton-LU", newton_lu),
            ("Newton-CG", newton_cg),
            ("Broyden", broyden),
        ):
            try:
                t0 = time.time()
                U_final, it = metodo(U0, tol, max_iter)
                t1 = time.time()

                erro = norma_inf(U_final - U_exata)
                resultados.append((N, nome, it, t1 - t0, erro))
                if nome == "Newton-LU":
                    erros_lu.append(erro)
                    if N_plot is None or N >= N_plot:
                        U_plot = U_final
                        U_exata_plot = U_exata
                        N_plot = N
            except Exception as exc:
                resultados.append((N, nome, None, None, None))
                print(f"Aviso: {nome} falhou em N={N}: {exc}")

    print("Resultados (N, Metodo, Iteracoes, Tempo(s), Erro Maximo)")
    print("-" * 68)
    for N, nome, it, tempo, erro in resultados:
        it_str = f"{it}" if it is not None else "-"
        tempo_str = f"{tempo:.6f}" if tempo is not None else "-"
        erro_str = f"{erro:.6e}" if erro is not None else "-"
        print(f"{N:>4}  {nome:<10}  {it_str:>6}  {tempo_str:>10}  {erro_str:>12}")

    print("\nOrdem de convergencia (Newton-LU)")
    print("-" * 40)
    for i in range(len(erros_lu) - 1):
        if erros_lu[i] is None or erros_lu[i + 1] is None:
            continue
        p = np.log(erros_lu[i] / erros_lu[i + 1]) / np.log(hs[i] / hs[i + 1])
        print(f"N={Ns[i]} -> N={Ns[i+1]}: p = {p:.4f}")

    if U_plot is not None and U_exata_plot is not None and N_plot is not None:
        plotar_resultados(U_plot, U_exata_plot, N_plot)
