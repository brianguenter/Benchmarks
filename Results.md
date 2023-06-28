# Comparison of FD with other AD algorithms
| Function | FD sparse | FD dense | ForwardDiff | ReverseDiff | Enzyme | Zygote |
|---------|-----------|----------|-------------|-------------|--------|--------|
| Rosenbrock Hessian | **1.00** | 60.00 | 603323.83 | 382581.09 | [^notes] | 966903.60 |
| Rosenbrock gradient | [^notes] | 1.18 | 606.36 | 270.80 | **1.00** | 3566.65 |
| Simple matrix Jacobian | [^notes] | **1.00** | 33.15 | 52.69 | [^notes] | 121.59 |
| Spherical harmonics Jacobian | [^notes] | **1.00** | 35.31 | [^notes] | [^notes] | [^notes] |


 ## Comparison of AD algorithms with a hand optimized Jacobian
| FD sparse | FD Dense | ForwardDiff | ReverseDiff | Enzyme | Zygote | Hand optimized|
|-----------|----------|-------------|-------------|--------|--------|---------------|
 **1.00** | 1.85 | 30.99 | [^notes] | [^notes] | 537196.68 | 2.48 |


It is worth nothing that both FD sparse and FD dense are faster than the hand optimized Jacobian.

[^notes]: For the FD sparse column, FD sparse was slower than FD dense so times are not listed for this column. For all other columns either the benchmark code crashes or I haven't yet figured out how to make it work correctly.
