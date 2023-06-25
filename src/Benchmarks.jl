module Benchmarks
import FastDifferentiation
import FastDifferentiation
import ForwardDiff
using BenchmarkTools
import ReverseDiff
using LinearAlgebra
import Enzyme
using Memoize
using StaticArrays

# include("ODE.jl")
include("SphericalHarmonics.jl")
include("ODE.jl")

const global n_size = 10

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

function ODE_comparison()
    y = FastDifferentiation.make_variables(:y, 20)
    dy = Vector{FastDifferentiation.Node}(undef, 20)
    ODE.f(dy, y, nothing, nothing)

    jac = FastDifferentiation.jacobian(dy, y)
    J = Matrix{FastDifferentiation.Node}(undef, 20, 20)
    ODE.fjac(J, y, nothing, nothing)
    println("number of operations for FD: $(FastDifferentiation.number_of_operations(jac))")
    println("number of operations for hand_jac: $(FastDifferentiation.number_of_operations(J))")

    fd_exe = FastDifferentiation.make_function(jac, y, in_place=true)
    float_J1 = Matrix{Float64}(undef, 20, 20)
    float_J2 = Matrix{Float64}(undef, 20, 20)
    float_y = rand(20)
    fd_exe(float_y, float_J1)

    sparse = FastDifferentiation.sparsity(FastDifferentiation.DerivativeGraph(dy))
    @info "sparsity of ODE $sparse"
    ODE.fjac(float_J2, float_y, nothing, nothing)
    @assert isapprox(float_J1, float_J2, atol=1e-11)

    return (@benchmark(ODE.fjac($float_J2, $float_y, nothing, nothing)), @benchmark $fd_exe($float_y, $float_J1))
end
export ODE_comparison


include("EnzymeBenchmarks.jl")
include("FDBenchmarks.jl")
include("ReverseDiffBenchmarks.jl")
include("ForwardDiffBenchmarks.jl")

# const hessian_benchmarks = (fd_rosenbrock_hessian, enzyme_rosenbrock_hessian, forward_diff_rosenbrock_hessian, reverse_diff_rosenbrock_hessian)
const rosenbrock_hessian_benchmarks = (fd_rosenbrock_hessian_sparse, fd_rosenbrock_hessian, forward_diff_rosenbrock_hessian, reverse_diff_rosenbrock_hessian)
export rosenbrock_hessian_benchmarks

# const jacobian_benchmarks = (
#     fd_rosenbrock_jacobian, enzyme_rosenbrock_gradient, forward_diff_rosenbrock_jacobian, reverse_diff_rosenbrock_jacobian,
#     fd_R¹⁰⁰R¹⁰⁰, forward_diff_R¹⁰⁰R¹⁰⁰, reverse_diff_R¹⁰⁰R¹⁰⁰)

const rosenbrock_jacobian_benchmarks = (fd_rosenbrock_jacobian, forward_diff_rosenbrock_jacobian, reverse_diff_rosenbrock_jacobian,)
export rosenbrock_jacobian_benchmarks

const R100_R100_jacobian_benchmarks = (fd_R¹⁰⁰R¹⁰⁰, forward_diff_R¹⁰⁰R¹⁰⁰, reverse_diff_R¹⁰⁰R¹⁰⁰)
export R100_R100_jacobian_benchmarks


function run_benchmarks(benchmarks, nterms)
    benches = [f(nterms) for f in benchmarks]
    fd_time = minimum(benches[1].times)
    times = [x !== nothing ? minimum(x.times) / fd_time : -1 for x in benches]

    return times
end
export run_benchmarks
end #module