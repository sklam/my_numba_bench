# Numba benchmarks

Uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/stable/).

Make environment with:

```bash
$ conda create --name <env> --file ./conda_env_export.txt
```

Run benchmark with:

```bash
$ pytest
```


Test groups:

* `compile`:
    Bencmark of the compiler
* `nrt`:
    Benchmark testing the numba runtime
* `cuda`:
    Bencmark of CUDA programs

See [pytest documentation on markers](http://doc.pytest.org/en/latest/example/markers.html) for details on how to use the test groups.
