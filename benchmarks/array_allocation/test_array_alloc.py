"""
Benchmark for array allocation
"""

import pytest
import numpy as np
from numba import njit


@njit
def allocate_many(output):
    for i in range(output.size):
        a = np.arange(i % 10)
        output[i] = a.sum()
    return output


@pytest.mark.parametrize("n", [1000, 100000])
def test_array_allocation(benchmark, n):
    # warm up
    allocate_many(np.zeros(10))
    # bench
    got = benchmark(lambda: allocate_many(output=np.zeros(n)))
    expect = allocate_many.py_func(np.zeros(n))
    assert np.all(got == expect)


def inspect():
    allocate_many(np.zeros(2))
    # allocate_many.inspect_cfg(allocate_many.signatures[0]).display(view=True)


if __name__ == "__main__":
    inspect()
