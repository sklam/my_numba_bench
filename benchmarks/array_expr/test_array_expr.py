import pytest
import numpy as np
from numba import njit, types


def array_expr(x):
    acc = 0
    acc2 = 0
    for i in range(len(x)):
        f = np.sqrt(x[i])
        acc += np.sin(x[i]) ** 2 + np.cos(x[i]) ** 2
        g = np.sqrt(f)
        acc2 += acc + g
    return acc2


@pytest.mark.parametrize("n", [100, 10000])
def test_array_expr_execution(benchmark, n):
    x = np.arange(n)
    jit_array_expr = njit(array_expr)
    assert jit_array_expr(x) == array_expr(x)
    benchmark(jit_array_expr, x=x)


@pytest.mark.parametrize("n", [10000])
def test_array_expr_parallel_execution(benchmark, n):
    x = np.arange(n)
    jit_array_expr = njit(parallel=True)(array_expr)
    assert np.allclose(jit_array_expr(x), array_expr(x))
    benchmark(jit_array_expr, x=x)


@pytest.mark.parametrize("n", [10000])
def test_array_expr_fastmath_execution(benchmark, n):
    x = np.arange(n)
    jit_array_expr = njit(fastmath=True)(array_expr)
    assert np.allclose(jit_array_expr(x), array_expr(x))
    benchmark(jit_array_expr, x=x)


@pytest.mark.compiler
def test_array_expr_compile(benchmark):
    def compile():
        jit_array_expr = njit(array_expr)
        sig = (types.float64[::1],)
        jit_array_expr.compile(sig)
        return jit_array_expr

    jit_array_expr = benchmark(compile)
    assert jit_array_expr.nopython_signatures


@pytest.mark.compiler
def test_array_expr_fastmath_compile(benchmark):
    def compile():
        jit_array_expr = njit(fastmath=True)(array_expr)
        sig = (types.float64[::1],)
        jit_array_expr.compile(sig)
        return jit_array_expr

    jit_array_expr = benchmark(compile)
    assert jit_array_expr.nopython_signatures


@pytest.mark.compiler
def test_array_expr_parallel_compile(benchmark):
    def compile():
        jit_array_expr = njit(parallel=True)(array_expr)
        sig = (types.float64[::1],)
        jit_array_expr.compile(sig)
        return jit_array_expr

    jit_array_expr = benchmark(compile)
    assert jit_array_expr.nopython_signatures
