## Results

These timings are just for evaluating the derivative function. They do not include preprocessing time required to generate and compile the function nor any time needed to generate auxiliary data structures that make the evaluation more efficient.

The times in each row are normalized to the shortest time in that row. The fastest algorithm will have a relative time of 1.0 and all other algorithms will have a time ≥ 1.0. Smaller numbers are better.

All benchmarks run on this system:
```julia 
Julia Version 1.9.2
Commit e4ee485e90 (2023-07-05 09:39 UTC)
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
  CPU: 32 × AMD Ryzen 9 7950X 16-Core Processor            
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 1 on 32 virtual cores
Environment:
  JULIA_EDITOR = code.cmd
``` 

| Function | FD sparse | FD dense | ForwardDiff | ReverseDiff | Enzyme | Zygote |
|---------|-----------|----------|-------------|-------------|--------|--------|
| Rosenbrock Hessian | **1.00** | 73.31 | 579092.33 | 440302.62 | [^5.2] | 1191965.23 |
| Rosenbrock gradient | [^1] | 1.29 | 683.59 | 305.32 | **1.00** | 4814.04 |
| Simple matrix Jacobian | [^1] | **1.00** | 48.10 | 48.83 | [^5] | 129.16 |
| Spherical harmonics Jacobian | [^1] | **1.00** | 35.28 | [^4] | [^5.1] | [^6] |
[^5.2]: fails with this error "ERROR: Function to differentiate is guaranteed to return an error and doesn't make sense to autodiff. Giving up"
[^1]: **FD** sparse was slower than **FD** dense so results are only shown for dense.
[^1]: **FD** sparse was slower than **FD** dense so results are only shown for dense.
[^5]: Enzyme prints "Warning: using fallback BLAS replacements, performance may be degraded", followed by stack overflow error or endless loop.
[^1]: **FD** sparse was slower than **FD** dense so results are only shown for dense.
[^4]: ReverseDiff failed on Spherical harmonics.
[^5.1]: Enzyme crashes Julia REPL for SHFunctions benchmark.
[^6]: Zygote doesn't work with Memoize


 ### Comparison of AD algorithms with a hand optimized Jacobian
This compares AD algorithms to a hand optimized Jacobian (in file ODE.jl). As before timings are relative to the fastest time.
Enzyme (array) is written to accept a vector input and return a matrix output to be compatible with the calling convention for the ODE function. This is very slow because Enzyme does not yet do full optimizations on the these input/output types. Enzyme (tuple) is written to accept a tuple input and returns tuple(tuples). This is much faster but not compatible with the calling convetions of the ODE function. This version uses features not avaialable in the registered version of Enzyme (as of 7-9-2023). You will need to `] add Enzyme#main` instead of using the registered version.

| FD sparse | FD Dense | ForwardDiff | ReverseDiff | Enzyme (array) | Enzyme (tuple) | Zygote | Hand optimized|
|-----------|----------|-------------|-------------|----------------|----------------|--------|---------------|
 **1.00** | 1.78 | 31.50 | [^4.1] | 323.85 | 4.31 | 561910.05 | 2.51 |


It is worth nothing that both FD sparse and FD dense are faster than the hand optimized Jacobian.
[^4.1]: ODE not implemented for ReverseDiff
