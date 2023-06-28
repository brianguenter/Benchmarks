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
using Printf


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

rosenbrock_jacobian_benchmarks = (fd_sparse_placeholder, fd_rosenbrock_gradient, forward_diff_rosenbrock_gradient, reverse_diff_rosenbrock_gradient, enzyme_rosenbrock_gradient, zygote_rosenbrock_gradient)
export rosenbrock_jacobian_benchmarks

R100_R100_jacobian_benchmarks = (fd_sparse_placeholder, fd_R¹⁰⁰R¹⁰⁰, forward_diff_R¹⁰⁰R¹⁰⁰, reverse_diff_R¹⁰⁰R¹⁰⁰, enzyme_R¹⁰⁰R¹⁰⁰, zygote_R¹⁰⁰R¹⁰⁰)
export R100_R100_jacobian_benchmarks

SH_Functions_benchmarks = (fd_sparse_placeholder, fd_SHFunctions,
    forward_diff_SHFunctions, reverse_diff_SHFunctions, enzyme_SHFunctions, zygote_SHFunctions)
export SH_Functions_benchmarks

ODE_benchmarks = (fd_ODE_sparse, fd_ODE, forward_diff_ODE, reverse_diff_ODE, enzyme_ODE, zygote_ODE, ODE.hand_ODE)
export ODE_benchmarks

function run_benchmarks(benchmarks, nterms=nothing)
    if nterms === nothing
        benches = [f() for f in benchmarks]
    else
        benches = [f(nterms) for f in benchmarks]
    end

    times = [x !== nothing ? minimum(x.times) : Inf for x in benches]
    min_time, _ = findmin(times)
    times /= min_time

    return times
end
export run_benchmarks

function run_all()
    all_benches = (rosenbrock_hessian_benchmarks, rosenbrock_jacobian_benchmarks, R100_R100_jacobian_benchmarks, SH_Functions_benchmarks)
    parameters = (1000, 1000, 10, 40)

    # for (bench, parameter) in zip(all_benches, parameters)
    #     println("benchmark $bench parameter $parameter\n")
    #     run_benchmarks(bench, parameter)
    # end
    return run_benchmarks.(all_benches, parameters)
end
export run_all

function write_timing(benchmark)
    one_row = ""

    fmt = "%.2f"

    for bench_times in benchmark
        _, min_index = findmin(bench_times)
        one_row *= "\n|"
        for (i, time) in pairs(bench_times)
            if time === Inf
                one_row *= " [^notes] |"
            else
                ftime = Printf.format(Printf.Format(fmt), time)
                if i == min_index
                    one_row *= " **$ftime** |"
                else
                    one_row *= " $ftime |"
                end
            end
        end
    end
    return one_row
end

function write_markdown()
    jacobian_header = """# Comparison of FD with other AD algorithms
    | Function | FD sparse | FD dense | ForwardDiff | ReverseDiff | Enzyme | Zygote |
    |---------|-----------|----------|-------------|-------------|--------|--------|"""



    for benchmark in run_all()
        jacobian_header *= write_timing(benchmark)
    end

    jacobian_header *= """\n\n ## Comparison of AD algorithms with a hand optimized Jacobian
    | FD sparse | FD Dense | ForwardDiff | ReverseDiff | Enzyme | Zygote | Hand optimized|
    |-----------|----------|-------------|-------------|--------|--------|---------------|"""

    benchmark = run_benchmarks(ODE_benchmarks)
    jacobian_header *= write_timing(benchmark)
    jacobian_header *= "\n\nIt is worth nothing that both FD sparse and FD dense are faster than the hand optimized Jacobian."

    jacobian_header *= "\n\n[^notes]: For the FD sparse column, FD sparse was slower than FD dense so times are not listed for this column. For all other columns either the benchmark code crashes or I haven't yet figured out how to make it work correctly.\n"
    io = open("Results.md", "w")
    try
        write(io, jacobian_header)
    catch
    finally
        close(io)
    end
end
export write_markdown

end #module
