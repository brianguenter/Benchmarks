### Results

To generate the markdown for the results in this section execute the function `write_markdown()` in the file `Benchmarks.jl`.
    
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
  JULIA_NUM_THREADS = 1
``` 

| Function | FD sparse | FD dense | ForwardDiff | ReverseDiff | Enzyme | Zygote |
|---------|-----------|----------|-------------|-------------|--------|--------|
| Rosenbrock Hessian | **1.00** | 15.60 | 67031.33 | 193591.71 | 367.77 | 163244.34 |
| Rosenbrock gradient | [^1] | 1.13 | 629.23 | 281.72 | **1.00** | 3967.02 |
| Simple matrix Jacobian | [^1] | **1.00** | 41.16 | 52.39 | [^50] | 123.91 |
| Spherical harmonics Jacobian | [^1] | **1.00** | 29.00 | [^40] | [^51] | [^6] |
[^1]: **FD** sparse was slower than **FD** dense so results are only shown for dense.
[^1]: **FD** sparse was slower than **FD** dense so results are only shown for dense.
[^50]: Enzyme prints "Warning: using fallback BLAS replacements, performance may be degraded", followed by stack overflow error or endless loop.
[^1]: **FD** sparse was slower than **FD** dense so results are only shown for dense.
[^40]: ReverseDiff failed on Spherical harmonics.
[^51]: Enzyme crashes Julia REPL for SHFunctions benchmark.
[^6]: Zygote doesn't work with Memoize


 ### Comparison to hand optimized Jacobian.
This compares AD algorithms to a hand optimized Jacobian (in file ODE.jl). As before timings are relative to the fastest time.
Enzyme (array) is written to accept a vector input and return a matrix output to be compatible with the calling convention for the ODE function. This is very slow because Enzyme does not yet do full optimizations on these input/output types. Enzyme (tuple) is written to accept a tuple input and returns tuple(tuples). This is much faster but not compatible with the calling convetions of the ODE function. This version uses features not avaialable in the registered version of Enzyme (as of 7-9-2023). You will need to `] add Enzyme#main` instead of using the registered version.

| FD sparse | FD Dense | ForwardDiff | ReverseDiff | Enzyme (array) | Enzyme (tuple) | Zygote | Hand optimized|
|-----------|----------|-------------|-------------|----------------|----------------|--------|---------------|
 **1.00** | 1.74 | 29.28 | [^41] | 255.63 | 4.22 | 504683.35 | 2.30 |


It is worth nothing that both FD sparse and FD dense are faster than the hand optimized Jacobian.
[^41]: ODE not implemented for ReverseDiff
