# linear_systems.py
"""Volume 1: Linear Systems.
<Name>
<Class>
<Date>
"""

import numpy as np
from time import perf_counter as time
from scipy import sparse
from scipy import linalg as la
from scipy.sparse import linalg as spla
from matplotlib import pyplot as plt


# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    #copy A
    A = A.astype(float).copy()
    #dimensions and stuff
    n = A.shape[0]
    #gonna go through the rows
    for i in range(n):
        pivot = A[i, i]
        #gets cols
        for j in range(i + 1, n):
            #row ops
            factor = A[j, i] / pivot
            A[j, i:] -= factor * A[i, i:]
    return A


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    #copy and initialize
    A = A.astype(float).copy()
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    #gets rows
    for i in range(n):
        pivot = U[i, i]
        #gets entries of that row
        for j in range(i + 1, n):
            #lu decomp
            factor = U[j, i] / pivot
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
    return L, U

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((n,) ndarray): The solution to the linear system.
    """
    #initialize
    L, U = lu(A)
    n = L.shape[0]
    y = np.zeros(n)

    #gonna back sub
    for i in range(n):
        y[i] = b[i] - np.sum(L[i,:i] * y[:i])
    x = np.zeros(n)

    for i in reversed(range(n)):
        x[i] = (y[i] - np.sum(U[i, i + 1:] * x[i + 1:])) / U[i, i]
    return x

def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    #we are going to time
    n_vals = [10, 50, 100, 200, 300, 400, 500, 1000]

    times_inv = []
    times_solve = []
    times_lu_full = []
    times_lu_solve_only =[]

    #makes your random matrix and b vect
    for n in n_vals:
        A = np.random.random((n,n))
        b = np.random.random((n))
        
        #gonna time
        start = time()
        A_inv = la.inv(A)
        x1 = A_inv @ b
        times_inv.append(time() - start)
        
        start = time()
        x2 = la.solve(A, b)
        times_solve.append(time() - start)

        start = time()
        lu, piv = la.lu_factor(A)
        x3 = la.lu_solve((lu, piv), b)
        times_lu_full.append(time() - start)

        start = time()
        lu, piv = la.lu_factor(A)
        
        x4 = la.lu_solve((lu, piv), b)
        times_lu_solve_only.append(time() - start)

    #this is the plot
    plt.figure(figsize=(10, 6))

    plt.plot(n_vals, times_inv, 'o-', label='A⁻¹b (inverse + multiply)', linewidth=2)
    plt.plot(n_vals, times_solve, 's-', label='la.solve()', linewidth=2)
    plt.plot(n_vals, times_lu_full, '^-', label='LU (factor + solve)', linewidth=2)
    plt.plot(n_vals, times_lu_solve_only, 'd-', label='LU (solve only)', linewidth=2)
    plt.tight_layout()
    plt.legend()
    plt.title("Time solving methods")
    plt.xlabel("Mat size")
    plt.ylabel("Time to solve")

    plt.savefig("linear_system_times.png")


# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    #construct the matrix
    main_diag = -4 * np.ones(n)
    upper_diag = np.ones(n-1)
    lower_diag = np.ones(n-1)
    B = sparse.diags([main_diag, upper_diag, lower_diag], [0,1,-1])
    I = sparse.eye(n, format = 'csr')

    #construct the sparse matrix
    blocks = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(B)
            elif abs(i - j) == 1:
                row.append(I)
            else:
                row.append(None)

        blocks.append(row)

    A = sparse.bmat(blocks, format = 'csr')
    return A

# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    
    #we will time the dense and sparse matrices
    ns = [5, 10, 20, 40, 80, 160]
    sparse_times = []
    dense_times = []
    sizes = []
    for n in ns:
        A = prob5(n)
        b = np.random.random(n**2)

        start = time()
        x_sparse = spla.spsolve(A, b)
        end = time()
        sparse_times.append(end - start)

        A_dense = A.toarray()
        start = time()
        x_dense = la.solve(A_dense, b)
        end = time()
        dense_times.append(end - start)

        sizes.append(n**2)

    #this is the fig
    
    plt.figure(figsize=(10, 4))

    plt.loglog(sizes, sparse_times, '-o', label='Sparse Solver')
    plt.loglog(sizes, dense_times, '-o', label='Dense Solver')
    plt.xlabel("System Size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.title("sparse matrices vs dense matrices")

    plt.savefig("sparse vs dense times.png")


if __name__ == "__main__":
    prob4()
    print("did 4")
    print(prob5(2))
    print("did 5")
    prob6()
    print("did 6")
