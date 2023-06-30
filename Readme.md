# Benchmark Problems



This is a set of benchmarks to compare FastDifferention (**FD**) to several other AD algorithms:
* ForwardDiff
* ReverseDiff
* Enzyme
* Zygote


These benchmarks are not complete either because I have not yet figured out how to compute the required derivative with that AD package or because I am not confident that I understand the package well enough to do it in the most efficient way. 

I believe the benchmarks reflect the best way to use each package. However, I am not an expert in any of these packages. The timings for Zygote seem unusually slow so it is possible it is not being used as efficiently as possible. If you are expert in any of these packages please submit a PR to fill in, improve, or correct a benchmark.

The benchmarks test the speed of gradients, Jacobians, Hessians, and the ability to exploit sparsity in the derivative. The last problem, `ODE`, also compares the AD algorithms to a hand optimized Jacobian.

When determining which AD algorithm to use keep in mind the limitations of **FD**. The total operation count of your expression should be less than 10⁵. You may get reasonable performance for expressions as large as 10⁶ operations but expect very long compile times. FD does not support conditionals which involve the differentiation variables (yet). The other algorithms do not have these limitations.

To get accurate results for the Enzyme benchmarks you must set the number of threads in Julia to 1. Otherwise Enzyme will generate slower thread safe code.


<details>
  <summary> Compute the gradient and the Hessian of the Rosenbrock function. The Hessian is extremely sparse so algorithms that can detect sparsity will have an advantage. </summary>

```
function rosenbrock(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b * (x[i+1] - x[i]^2)^2
    end
    return result
end
export rosenbrock
```
</details>


<details> 
    <summary> Compute the Jacobian of a small matrix function. </summary>

```
   f(a, b) = (a + b) * (a * b)'
```
</details>

<details> 
    <summary> Compute the Jacobian of SHFunctions which constructs the spherical harmonics of order `n`: </summary>

```

@memoize function P(l, m, z)
    if l == 0 && m == 0
        return 1.0
    elseif l == m
        return (1 - 2m) * P(m - 1, m - 1, z)
    elseif l == m + 1
        return (2m + 1) * z * P(m, m, z)
    else
        return ((2l - 1) / (l - m) * z * P(l - 1, m, z) - (l + m - 1) / (l - m) * P(l - 2, m, z))
    end
end
export P

@memoize function S(m, x, y)
    if m == 0
        return 0
    else
        return x * C(m - 1, x, y) - y * S(m - 1, x, y)
    end
end
export S

@memoize function C(m, x, y)
    if m == 0
        return 1
    else
        return x * S(m - 1, x, y) + y * C(m - 1, x, y)
    end
end
export C

function factorial_approximation(x)
    local n1 = x
    sqrt(2 * π * n1) * (n1 / ℯ * sqrt(n1 * sinh(1 / n1) + 1 / (810 * n1^6)))^n1
end
export factorial_approximation

function compare_factorial_approximation()
    for n in 1:30
        println("n $n relative error $((factorial(big(n))-factorial_approximation(n))/factorial(big(n)))")
    end
end
export compare_factorial_approximation

@memoize function N(l, m)
    @assert m >= 0
    if m == 0
        return sqrt((2l + 1 / (4π)))
    else
        # return sqrt((2l+1)/2π * factorial(big(l-m))/factorial(big(l+m)))
        #use factorial_approximation instead of factorial because the latter does not use Stirlings approximation for large n. Get error for n > 2 unless using BigInt but if use BigInt get lots of rational numbers in symbolic result.
        return sqrt((2l + 1) / 2π * factorial_approximation(l - m) / factorial_approximation(l + m))
    end
end
export N

"""l is the order of the spherical harmonic"""
@memoize function Y(l, m, x, y, z)
    @assert l >= 0
    @assert abs(m) <= l
    if m < 0
        return N(l, abs(m)) * P(l, abs(m), z) * S(abs(m), x, y)
    else
        return N(l, m) * P(l, m, z) * C(m, x, y)
    end
end
export Y

function SHFunctions(max_l, x::T, y::T, z::T) where {T}
    shfunc = Vector{T}(undef, max_l^2)
    for l in 0:max_l-1
        for m in -l:l
            push!(shfunc, Y(l, m, x, y, z))
        end
    end

    return shfunc
end

function SHFunctions(max_l, x::FastDifferentiation.Node, y::FastDifferentiation.Node, z::FastDifferentiation.Node)
    shfunc = FastDifferentiation.Node[]

    for l in 0:max_l-1
        for m in -l:l
            push!(shfunc, (Y(l, m, x, y, z)))
        end
    end

    return shfunc
end
export SHFunctions
```

</details>

<details> 
    <summary> Compute the 20x20 Jacobian, ∂dy/∂y, of this function (used in an ODE problem) and compare to a hand optimized Jacobian. The Jacobian is approximately 25% non-zeros so algorithms that exploit sparsity in the derivative will have an advantage. </summary>

```

const k1 = .35e0
const k2 = .266e2
const k3 = .123e5
const k4 = .86e-3
const k5 = .82e-3
const k6 = .15e5
const k7 = .13e-3
const k8 = .24e5
const k9 = .165e5
const k10 = .9e4
const k11 = .22e-1
const k12 = .12e5
const k13 = .188e1
const k14 = .163e5
const k15 = .48e7
const k16 = .35e-3
const k17 = .175e-1
const k18 = .1e9
const k19 = .444e12
const k20 = .124e4
const k21 = .21e1
const k22 = .578e1
const k23 = .474e-1
const k24 = .178e4
const k25 = .312e1

function f(dy, y, p, t)
    r1 = k1 * y[1]
    r2 = k2 * y[2] * y[4]
    r3 = k3 * y[5] * y[2]
    r4 = k4 * y[7]
    r5 = k5 * y[7]
    r6 = k6 * y[7] * y[6]
    r7 = k7 * y[9]
    r8 = k8 * y[9] * y[6]
    r9 = k9 * y[11] * y[2]
    r10 = k10 * y[11] * y[1]
    r11 = k11 * y[13]
    r12 = k12 * y[10] * y[2]
    r13 = k13 * y[14]
    r14 = k14 * y[1] * y[6]
    r15 = k15 * y[3]
    r16 = k16 * y[4]
    r17 = k17 * y[4]
    r18 = k18 * y[16]
    r19 = k19 * y[16]
    r20 = k20 * y[17] * y[6]
    r21 = k21 * y[19]
    r22 = k22 * y[19]
    r23 = k23 * y[1] * y[4]
    r24 = k24 * y[19] * y[1]
    r25 = k25 * y[20]

    dy[1] = -r1 - r10 - r14 - r23 - r24 +
            r2 + r3 + r9 + r11 + r12 + r22 + r25
    dy[2] = -r2 - r3 - r9 - r12 + r1 + r21
    dy[3] = -r15 + r1 + r17 + r19 + r22
    dy[4] = -r2 - r16 - r17 - r23 + r15
    dy[5] = -r3 + r4 + r4 + r6 + r7 + r13 + r20
    dy[6] = -r6 - r8 - r14 - r20 + r3 + r18 + r18
    dy[7] = -r4 - r5 - r6 + r13
    dy[8] = r4 + r5 + r6 + r7
    dy[9] = -r7 - r8
    dy[10] = -r12 + r7 + r9
    dy[11] = -r9 - r10 + r8 + r11
    dy[12] = r9
    dy[13] = -r11 + r10
    dy[14] = -r13 + r12
    dy[15] = r14
    dy[16] = -r18 - r19 + r16
    dy[17] = -r20
    dy[18] = r20
    dy[19] = -r21 - r22 - r24 + r23 + r25
    dy[20] = -r25 + r24
end
```

</details>

<details>
    <summary> This is the hand optimized Jacobian, ∂dy/∂y, from the ODE function, above. Your Jacobian function should zero out the in place array J which the Jacobian result will be written into, unless you are using sparse arrays. </summary>

```
function fjac(J, y, p, t)
    J .= zero(eltype(J))
    J[1, 1] = -k1 - k10 * y[11] - k14 * y[6] - k23 * y[4] - k24 * y[19]
    J[1, 11] = -k10 * y[1] + k9 * y[2]
    J[1, 6] = -k14 * y[1]
    J[1, 4] = -k23 * y[1] + k2 * y[2]
    J[1, 19] = -k24 * y[1] + k22
    J[1, 2] = k2 * y[4] + k9 * y[11] + k3 * y[5] + k12 * y[10]
    J[1, 13] = k11
    J[1, 20] = k25
    J[1, 5] = k3 * y[2]
    J[1, 10] = k12 * y[2]

    J[2, 4] = -k2 * y[2]
    J[2, 5] = -k3 * y[2]
    J[2, 11] = -k9 * y[2]
    J[2, 10] = -k12 * y[2]
    J[2, 19] = k21
    J[2, 1] = k1
    J[2, 2] = -k2 * y[4] - k3 * y[5] - k9 * y[11] - k12 * y[10]

    J[3, 1] = k1
    J[3, 4] = k17
    J[3, 16] = k19
    J[3, 19] = k22
    J[3, 3] = -k15

    J[4, 4] = -k2 * y[2] - k16 - k17 - k23 * y[1]
    J[4, 2] = -k2 * y[4]
    J[4, 1] = -k23 * y[4]
    J[4, 3] = k15

    J[5, 5] = -k3 * y[2]
    J[5, 2] = -k3 * y[5]
    J[5, 7] = 2k4 + k6 * y[6]
    J[5, 6] = k6 * y[7] + k20 * y[17]
    J[5, 9] = k7
    J[5, 14] = k13
    J[5, 17] = k20 * y[6]

    J[6, 6] = -k6 * y[7] - k8 * y[9] - k14 * y[1] - k20 * y[17]
    J[6, 7] = -k6 * y[6]
    J[6, 9] = -k8 * y[6]
    J[6, 1] = -k14 * y[6]
    J[6, 17] = -k20 * y[6]
    J[6, 2] = k3 * y[5]
    J[6, 5] = k3 * y[2]
    J[6, 16] = 2k18

    J[7, 7] = -k4 - k5 - k6 * y[6]
    J[7, 6] = -k6 * y[7]
    J[7, 14] = k13

    J[8, 7] = k4 + k5 + k6 * y[6]
    J[8, 6] = k6 * y[7]
    J[8, 9] = k7

    J[9, 9] = -k7 - k8 * y[6]
    J[9, 6] = -k8 * y[9]

    J[10, 10] = -k12 * y[2]
    J[10, 2] = -k12 * y[10] + k9 * y[11]
    J[10, 9] = k7
    J[10, 11] = k9 * y[2]

    J[11, 11] = -k9 * y[2] - k10 * y[1]
    J[11, 2] = -k9 * y[11]
    J[11, 1] = -k10 * y[11]
    J[11, 9] = k8 * y[6]
    J[11, 6] = k8 * y[9]
    J[11, 13] = k11

    J[12, 11] = k9 * y[2]
    J[12, 2] = k9 * y[11]

    J[13, 13] = -k11
    J[13, 11] = k10 * y[1]
    J[13, 1] = k10 * y[11]

    J[14, 14] = -k13
    J[14, 10] = k12 * y[2]
    J[14, 2] = k12 * y[10]

    J[15, 1] = k14 * y[6]
    J[15, 6] = k14 * y[1]

    J[16, 16] = -k18 - k19
    J[16, 4] = k16

    J[17, 17] = -k20 * y[6]
    J[17, 6] = -k20 * y[17]

    J[18, 17] = k20 * y[6]
    J[18, 6] = k20 * y[17]

    J[19, 19] = -k21 - k22 - k24 * y[1]
    J[19, 1] = -k24 * y[19] + k23 * y[4]
    J[19, 4] = k23 * y[1]
    J[19, 20] = k25

    J[20, 20] = -k25
    J[20, 1] = k24 * y[19]
    J[20, 19] = k24 * y[1]

    return nothing
end
```
</details>



## Comparison of FD with other AD algorithms


These timings are just for evaluating the derivative function. They do not include preprocessing time to generate either the function or auxiliary data structures that make the evaluation more efficient.

The times in each row are normalized to the shortest time in that row. The fastest algorithm will have a relative time of 1.0 and all other algorithms will have a time ≥ 1.0. Smaller numbers are better.

| Function | FD sparse | FD dense | ForwardDiff | ReverseDiff | Enzyme | Zygote |
|---------|-----------|----------|-------------|-------------|--------|--------|
| Rosenbrock Hessian | **1.00** | 75.60 | 571669.52 | 423058.61 | [^a] | 1015635.96 |
| Rosenbrock gradient | [^a] | 1.28 | 682.41 | 306.27 | **1.00** | 4726.62 |
| Simple matrix Jacobian | [^a] | **1.00** | 42.61 | 54.60 | [^a] | 130.13 |
| Spherical harmonics Jacobian | [^a] | **1.00** | 36.00 | [^a] | [^a] | [^a] |


 ### Comparison of AD algorithms with a hand optimized Jacobian
This compares AD algorithms to a hand optimized Jacobian (in file ODE.jl)
| FD sparse | FD Dense | ForwardDiff | ReverseDiff | Enzyme | Zygote | Hand optimized|
|-----------|----------|-------------|-------------|--------|--------|---------------|
 **1.00** | 1.81 | 29.45 | [^a] | [^a] | 556889.67 | 2.47 |


It is worth nothing that both FD sparse and FD dense are faster than the hand optimized Jacobian.

It is also intersting to note the ratio of the number of operations of the FD Jacobian of a function to the number of operations in the original function. 

Problem sizes in approximately the ratio 1:10:100:1000 were computed for several of the benchmarks. The parameters which give these ratios were: ((10,4,2),(100,11,4),(1000,35,9),(\_,\_,20)) for (Rosenbrock Jacobian, Spherical harmonics Jacobian, Simple matrix ops Jacobian), respectively. 

The ratio (jacobian operations)/(original function operations) stays close to a constant over 2 orders of magnitude of problem size for Rosenbrock and Spherical harmonics. For the simple matrix ops Jacobian the ratio goes from 2.6 to 6.5 over 3 orders of magnitude of problem size. This is an increase of 2.5x. But the smallest instance is an R⁸->R⁴ function and the largest is R⁸⁰⁰->R⁴⁰⁰ an increase in dimensions of a factor of 100x.

|Relative problem size | Rosenbrock Jacobian | Spherical harmonics Jacobian | Simple matrix ops Jacobian |
|-------|---------------------|------------------------------|------------------------|
|  1x     | 1.13                | 2.2                          |          2.6           |
|  10x     | 1.13                | 2.34                          |          3.5          |
|  100x     | 1.13                | 2.4                          |          3.8          |
| 1000x     |                      |                             |          6.5          |

This is a very small sample of functions but it will be interesting to see if this slow growth of the Jacobian with  increasing domain and codomain dimensions generalizes to all functions or only applies to functions with special graph structure.

[^a]: For the FD sparse column, FD sparse was slower than FD dense so times are not listed for this column. For all other columns either the benchmark code crashes or I haven't yet figured out how to make it work correctly and efficiently.
