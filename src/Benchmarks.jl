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
using SparseArrays
using InteractiveUtils


# include("ODE.jl")
include("SphericalHarmonics.jl")
include("ODE.jl")
include("FDProduct.jl")

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
    sparse_fd_exe(float_y, sparse_J)

    sparse = FastDifferentiation.sparsity(FastDifferentiation.DerivativeGraph(dy))
    @info "sparsity of ODE $sparse"
    ODE.fjac(float_J2, float_y, nothing, nothing)
    @assert isapprox(float_J1, float_J2, atol=1e-11)
    @assert isapprox(float_J2, sparse_J)

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

R100_R100_JACOBIAN_BENCHMARKS = (fd_sparse_placeholder, fd_R¹⁰⁰R¹⁰⁰, forward_diff_R¹⁰⁰R¹⁰⁰, reverse_diff_R¹⁰⁰R¹⁰⁰, enzyme_R¹⁰⁰R¹⁰⁰, zygote_R¹⁰⁰R¹⁰⁰)
export R100_R100_JACOBIAN_BENCHMARKS

SH_FUNCTIONS_BENCHMARKS = (fd_sparse_placeholder, fd_SHFunctions,
    forward_diff_SHFunctions, reverse_diff_SHFunctions, enzyme_SHFunctions, zygote_SHFunctions)
export SH_FUNCTIONS_BENCHMARKS

ODE_BENCHMARKS = (fd_ODE_sparse, fd_ODE, forward_diff_ODE, reverse_diff_ODE, enzyme_ODE, enzyme_tuple_ODE, zygote_ODE, ODE.hand_ODE)
export ODE_BENCHMARKS

ALL_BENCHMARKS = (rosenbrock_hessian_benchmarks, rosenbrock_jacobian_benchmarks, R100_R100_JACOBIAN_BENCHMARKS, SH_FUNCTIONS_BENCHMARKS)
export ALL_BENCHMARKS

all_names = ("Rosenbrock Hessian", "Rosenbrock gradient", "Simple matrix Jacobian", "Spherical harmonics Jacobian")

function run_benchmarks(benchmarks, nterms=nothing)
    @info "Running benchmark $benchmarks"
    if nterms === nothing
        benches = [f() for f in benchmarks]
    else
        benches = [f(nterms) for f in benchmarks]
    end

    times = [x isa Tuple ? x : minimum(x.times) for x in benches]
    min_time, _ = findmin(x -> x isa Tuple ? Inf : x, times)

    return map(x -> x isa Tuple ? x : x / min_time, times)
end
export run_benchmarks

function run_all(bench_list)

    parameters = (200, 1000, 10, 40)

    # for (bench, parameter) in zip(all_benches, parameters)
    #     println("benchmark $bench parameter $parameter\n")
    #     run_benchmarks(bench, parameter)
    # end
    return run_benchmarks.(bench_list, parameters)
end
export run_all

function write_timing(benchmark)
    one_row = ""

    fmt = "%.2f"

    footnotes = String[]

    _, min_index = findmin(x -> x isa Tuple ? Inf : x, benchmark)
    for (i, time) in pairs(benchmark)
        if time isa Tuple
            one_row *= " $(time[1]) |"
            println("before append $(time[2])")
            push!(footnotes, time[2])
            println("after append")
        else
            ftime = Printf.format(Printf.Format(fmt), time)
            if i == min_index
                one_row *= " **$ftime** |"
            else
                one_row *= " $ftime |"
            end
        end
    end
    one_row *= "\n"



    return one_row, footnotes
end
export write_timing

function test_data()
    return (collect(1:7), collect(8:14), collect(15:21)), collect(1:7)
end
export test_data

function write_markdown()
    benchmark_times = run_all(ALL_BENCHMARKS)
    ODE_times = run_benchmarks(ODE_BENCHMARKS)
    write_markdown(benchmark_times, all_names, ODE_times)
end

function write_markdown(benchmark_times, function_names, ODE_times)
    io = open("Results.md", "w")
    try
        write(
            io,
            """### Results

    To generate the markdown for the results in this section execute the function `write_markdown()` in the file `Benchmarks.jl`.
        
    These timings are just for evaluating the derivative function. They do not include preprocessing time required to generate and compile the function nor any time needed to generate auxiliary data structures that make the evaluation more efficient.

    The times in each row are normalized to the shortest time in that row. The fastest algorithm will have a relative time of 1.0 and all other algorithms will have a time ≥ 1.0. Smaller numbers are better.

    All benchmarks run on this system:
    """
        )

        write(io, "```julia \n")
        InteractiveUtils.versioninfo(io)
        write(io, "``` \n")

        write(
            io,
            """

| Function | FD sparse | FD dense | ForwardDiff | ReverseDiff | Enzyme | Zygote |
|---------|-----------|----------|-------------|-------------|--------|--------|\n"""
        )



        footnotes = String[]
        for (name, benchmark) in zip(function_names, benchmark_times)
            println("Benchmark $name $benchmark")
            entry, footnote = write_timing(benchmark)
            append!(footnotes, footnote)
            println("footnote $footnote")
            println("writing entry $entry")
            write(io, "| $name |" * entry)
        end

        for note in footnotes
            println("writing footnote $note")
            write(io, note * "\n")
        end
        write(
            io,
            """\n\n ### Comparison to hand optimized Jacobian.
   This compares AD algorithms to a hand optimized Jacobian (in file ODE.jl). As before timings are relative to the fastest time.
   Enzyme (array) is written to accept a vector input and return a matrix output to be compatible with the calling convention for the ODE function. This is very slow because Enzyme does not yet do full optimizations on these input/output types. Enzyme (tuple) is written to accept a tuple input and returns tuple(tuples). This is much faster but not compatible with the calling convetions of the ODE function. This version uses features not avaialable in the registered version of Enzyme (as of 7-9-2023). You will need to `] add Enzyme#main` instead of using the registered version.

   | FD sparse | FD Dense | ForwardDiff | ReverseDiff | Enzyme (array) | Enzyme (tuple) | Zygote | Hand optimized|
   |-----------|----------|-------------|-------------|----------------|----------------|--------|---------------|\n"""
        )

        row, footnotes = write_timing(ODE_times)
        write(io, row)
        write(io, "\n\nIt is worth nothing that both FD sparse and FD dense are faster than the hand optimized Jacobian.\n")

        for note in footnotes
            write(io, note * "\n")
        end
    catch e
        @info "$e error writing file"
    finally
        close(io)
    end
end
export write_markdown

function compare_operation_count(ros_size, sh_size, mat_size)
    xvec = FastDifferentiation.make_variables(:x, ros_size)
    FastDifferentiation.@variables x y z

    orig_rosenbrock = FastDifferentiation.number_of_operations([rosenbrock(xvec)])
    jac_rosenbrock = FastDifferentiation.number_of_operations(FastDifferentiation.jacobian([rosenbrock(xvec)], xvec))

    println("orign ops $orig_rosenbrock ratio Jacobian of rosenbrock to original $(jac_rosenbrock/orig_rosenbrock)")

    orig_SHFunctions = FastDifferentiation.number_of_operations(SHFunctions(sh_size, x, y, z))
    jac_SH = FastDifferentiation.number_of_operations(FastDifferentiation.jacobian(SHFunctions(sh_size, x, y, z), [x, y, z]))

    println("orig ops $orig_SHFunctions ratio Jacobian of SHFunctions to original $(jac_SH/orig_SHFunctions)")

    f(a, b) = (a + b) * (a * b)'

    av = FastDifferentiation.make_variables(:ain, mat_size^2)
    bv = FastDifferentiation.make_variables(:bin, mat_size^2)

    ain = reshape(av, mat_size, mat_size)
    bin = reshape(bv, mat_size, mat_size)

    orig_R_func = vec(FastDifferentiation.Node.(f(ain, bin)))
    orig_R = FastDifferentiation.number_of_operations(orig_R_func)
    inputs = vcat(av, bv)

    jac_R = FastDifferentiation.number_of_operations(FastDifferentiation.jacobian(orig_R_func, inputs))

    println("orig ops $orig_R ratio Jacobian of matrix function to original $(jac_R/orig_R)")

    yvec = FastDifferentiation.make_variables(:y, 20)
    dy = Vector{FastDifferentiation.Node}(undef, 20)
    ODE.f(dy, yvec, nothing, nothing)
    orig_ops = FastDifferentiation.number_of_operations(dy)
    jac_hand = FastDifferentiation.number_of_operations(FastDifferentiation.jacobian(dy, yvec))

    println("ratio Jacobian of hand optimized to original $(jac_hand/orig_ops)")


end
export compare_operation_count

end #module
