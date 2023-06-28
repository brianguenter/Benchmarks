module Benchmarks
import FastDifferentiation
import FastDifferentiation
import ForwardDiff
using BenchmarkTools
import ReverseDiff
import Zygote
using LinearAlgebra
import Enzyme
using Memoize
using StaticArrays

# include("ODE.jl")
include("SphericalHarmonics.jl")
include("ODE.jl")

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

    sparse_jac = FastDifferentiation.sparse_jacobian(dy, y)
    sparse_fd_exe = FastDifferentiation.make_function(sparse_jac, y, in_place=true)
    sparse_J = similar(sparse_jac, Float64)
    println("sparse $(sparse_fd_exe(float_y, sparse_J))")

    sparse = FastDifferentiation.sparsity(FastDifferentiation.DerivativeGraph(dy))
    @info "sparsity of ODE $sparse"
    ODE.fjac(float_J2, float_y, nothing, nothing)
    @assert isapprox(float_J1, float_J2, atol=1e-11)


    a = Any[]
    push!(a, @benchmark $sparse_fd_exe($float_y, $sparse_J))
    push!(a, @benchmark $fd_exe($float_y, $float_J1))
    push!(a, @benchmark(ODE.fjac($float_J2, $float_y, nothing, nothing)))
    times = map(x -> minimum(x.times), a)
    return a, map(x -> x / times[1], times)
end
export ODE_comparison


include("EnzymeBenchmarks.jl")
include("FDBenchmarks.jl")
include("ReverseDiffBenchmarks.jl")
include("ForwardDiffBenchmarks.jl")
include("ZygoteBenchmarks.jl")


rosenbrock_hessian_benchmarks = (fd_rosenbrock_hessian_sparse, fd_rosenbrock_hessian, forward_diff_rosenbrock_hessian, reverse_diff_rosenbrock_hessian, enzyme_rosenbrock_hessian, zygote_rosenbrock_hessian)
export rosenbrock_hessian_benchmarks

rosenbrock_jacobian_benchmarks = (fd_rosenbrock_gradient, forward_diff_rosenbrock_gradient, reverse_diff_rosenbrock_gradient, enzyme_rosenbrock_gradient, zygote_rosenbrock_gradient)
export rosenbrock_jacobian_benchmarks

R100_R100_jacobian_benchmarks = (fd_R¹⁰⁰R¹⁰⁰, forward_diff_R¹⁰⁰R¹⁰⁰, reverse_diff_R¹⁰⁰R¹⁰⁰, enzyme_R¹⁰⁰R¹⁰⁰, zygote_R¹⁰⁰R¹⁰⁰)
export R100_R100_jacobian_benchmarks

SH_Functions_benchmarks = (fd_SHFunctions,
    forward_diff_SHFunctions, reverse_diff_SHFunctions, enzyme_SHFunctions, ODE.no_function)
export SH_Functions_benchmarks

ODE_benchmarks = (fd_ODE_sparse, fd_ODE, forward_diff_ODE, reverse_diff_ODE, enzyme_ODE, zygote_ODE)
export ODE_benchmarks

function run_benchmarks(benchmarks, nterms=nothing)
    if nterms === nothing
        benches = [f() for f in benchmarks]
    else
        benches = [f(nterms) for f in benchmarks]
    end
    fd_time = minimum(benches[1].times)
    times = [x !== nothing ? minimum(x.times) / fd_time : missing for x in benches]

    return times
end
export run_benchmarks

function run_all()
    all_benches = (rosenbrock_hessian_benchmarks, rosenbrock_jacobian_benchmarks, R100_R100_jacobian_benchmarks, SH_Functions_benchmarks, ODE_benchmarks)
    parameters = (1000, 1000, 10, 40, nothing)

    return run_benchmarks.(all_benches, parameters)
end
export run_all

end #module
