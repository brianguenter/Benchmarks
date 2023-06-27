| Function                               | FD sparse    | FD dense      | ForwardDiff | ReverseDiff | Enzyme | Hand optimized Jacobian|
| - | - | - |- |- |- |
|Rosenbrock Hessian     | 1.0  | 75       | 576,276 | 454,145 | missing |missing |
| Rosenbrock Gradient      | [^1]  |1.0       | 530 |237| **0.87** |missing |
| Small Matrix Jacobian   |[^1]     | 1.0 |35|51.5|missing|missing |
| Spherical Harmonics Jacobian        | [^1]           | 1.0    | 37 |missing | missing |missing |
| ODE | 1.0 | 1.75 | 32 | missing | missing |

Fastest time for a benchmark is shown in **bold**.

For the hand optimized Jacobian FD

[^1]:No benefit from using FD Sparse


([1.0, 75.72289156626506, 576276.0240963856, 454145.7228915663], 
[1.0, 530.8344298245614, 237.6217105263158, 0.8715117872807018], 
Union{Missing, Float64}[1.0, 35.66329966329966, 51.15151515151515, missing], 
Union{Missing, Float64}[1.0, 37.40963855421687, missing, missing], 
Union{Missing, Float64}[1.0, 1.7505107957369264, 32.224532224532226, missing, missing])