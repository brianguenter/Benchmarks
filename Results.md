Times are normalized to FD sparse or FD Dense, whichever is smaller. 

Fastest normalized time for a benchmark is shown in **bold**. 

Smaller numbers are better.

| Function                               | FD sparse    | FD dense      | ForwardDiff | ReverseDiff | Enzyme | Hand optimized Jacobian|
| - | - | - |- |- |- | -  |
|Rosenbrock Hessian     | 1.0  | 75       | 576,276 | 454,145 | missing | [^2] |
| Rosenbrock Gradient      | [^1]  |1.0       | 530 |237| **0.87** |[^2] |
| Small Matrix Jacobian   |[^1]     | 1.0 |35|51.5|missing|[^2] |
| Spherical Harmonics Jacobian        | [^1]           | 1.0    | 37 |missing | missing |[^2] |
| ODE Jacobian | 1.0 | 1.8 | 32 | missing | missing | 2.47 |



It is worth nothing that for the ODE benchmark **FD** sparse and **FD** dense are both faster than the hand optimized Jacobian.

[^1]: No benefit from using FD Sparse
[^2]: No hand optmized Jacobian for these functions

