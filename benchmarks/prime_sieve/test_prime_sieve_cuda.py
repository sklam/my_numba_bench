"""
Implement Sieve of Eratosthenes for finding prime.
"""

import pytest
import numpy as np
from numba import cuda
from numba import types


STATE_UNSET = 0
STATE_ISPRIME = 1
STATE_NOTPRIME = 2


def get_sieve_prime():
    """Returns a implementation of the sieve_prime to force recompilation.
    """

    @cuda.jit
    def sieve_prime(num_states, base_prime):
        thread = cuda.grid(1)
        thread_stride = cuda.gridsize(1)
        for i in range(thread, num_states.size, thread_stride):
            x = i + base_prime + 1

            state = num_states[i]
            if state == STATE_UNSET:
                if x % base_prime == 0:  # is divisible?
                    num_states[i] = STATE_NOTPRIME

    return sieve_prime


def run(n, verbose=False, sieve_prime=None):
    sieve_prime = sieve_prime or get_sieve_prime()

    orig_num_states = num_states = np.zeros(n, dtype=np.uint8)
    # offset to the index of `num_states`
    first_num = 2  # start with 1st prime

    base_prime = first_num
    num_states[0] = STATE_ISPRIME

    num_states = num_states[1:]

    while num_states.size:
        if verbose:
            print(base_prime, "is prime")

        sieve_prime.forall(num_states.size)(num_states, base_prime)

        # Search next prime
        for i in range(num_states.size):
            if num_states[i] == STATE_UNSET:
                num_states[i] = STATE_ISPRIME
                num_states = num_states[i + 1 :]
                base_prime += i + 1
                break
        else:
            break

    primes = np.nonzero(orig_num_states == STATE_ISPRIME)[0] + first_num
    if verbose:
        print(primes)
    return primes


@pytest.mark.cuda
@pytest.mark.parametrize("n", [500, 1000, 5000])
def test_prime_sieve_cuda_execute(benchmark, n):
    """Measure execution time prime sieve
    """
    expected = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
    ]
    # warm up and verify
    sieve_prime = get_sieve_prime()
    got = run(n=120, sieve_prime=sieve_prime)
    np.testing.assert_equal(got, expected)
    # benchmark
    benchmark(run, n=n, sieve_prime=sieve_prime)


@pytest.mark.cuda
@pytest.mark.compiler
def test_prime_sieve_cuda_compilation(benchmark):
    """Measure compile time for prime_sieve kernel
    """
    sig = (types.uint8[::1], types.intp)

    def compile():
        sieve_prime = get_sieve_prime()
        sieve_prime.compile(sig)
        # make sure the PTX is available
        sieve_prime.inspect_asm(sig)

    benchmark(compile)


if __name__ == "__main__":
    run(n=120, verbose=True)
