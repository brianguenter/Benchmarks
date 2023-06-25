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


function ODE_comparison()
    y = FastDifferentiation.make_variables(:y, 20)
    dy = Vector{FastDifferentiation.Node}(undef, 20)
    ODE.f(dy, y, nothing, nothing)

    jac = FastDifferentiation.jacobian(dy, y)
    J = Matrix{FastDifferentiation.Node}(undef, 20, 20)
    ODE.fjac(J, y, nothing, nothing)
    println("number of operations for FD: $(FastDifferentiation.number_of_operations(jac))")
    println("number of operations for hand_jac: $(FastDifferentiation.number_of_operations(J))")

    fd_jac = FastDifferentiation.make_function(jac, y, in_place=true)
    float_J1 = Matrix{Float64}(undef, 20, 20)
    float_J2 = Matrix{Float64}(undef, 20, 20)
    float_y = rand(20)
    fd_jac(float_y, float_J1)

    ODE.fjac(float_J2, float_y, nothing, nothing)
    @assert isapprox(float_J1, float_J2, atol=1e-11)

    return (@benchmark(ODE.fjac($float_J2, $float_y, nothing, nothing)), @benchmark $fd_jac($float_y, $float_J1))
end
export ODE_comparison

# let's use a Rosenbrock function as our target function
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

"""This function pairs with `time_fd_rosenbrock`"""
function forward_diff_rosenbrock_jacobian(nterms)
    x = rand(nterms)

    # output buffer
    out = similar(x)
    fastest = typemax(Float64)
    local best_trial

    for chunk_size in 1:3:min(nterms, 10)
        cfg = ForwardDiff.GradientConfig(rosenbrock, x, ForwardDiff.Chunk{chunk_size}())
        trial = @benchmark ForwardDiff.gradient!($out, $rosenbrock, $x, $cfg)
        if minimum(trial).time < fastest
            best_trial = trial
            fastest = minimum(trial).time
        end
    end
    return best_trial
end
export forward_diff_rosenbrock_jacobian

function forward_diff_R¹⁰⁰R¹⁰⁰(nsize)
end

function forward_diff_rosenbrock_hessian(nterms)
end

function forward_diff_SHFunctions(nterms)
    x = rand(3)

    out = similar(x, (nterms^2, 3))
    fastest = typemax(Float64)
    local best_trial
    sh_wrapper(x) = SHFunctions(nterms, x[1], x[2], x[3])

    for chunk_size in 1:1:3
        cfg = ForwardDiff.JacobianConfig(sh_wrapper, x, ForwardDiff.Chunk{chunk_size}())
        trial = @benchmark ForwardDiff.jacobian($sh_wrapper, $x, $cfg)
        if minimum(trial).time < fastest
            best_trial = trial
            fastest = minimum(trial).time
        end
    end
    return best_trial
end
export forward_diff_SHFunctions

function fd_rosenbrock_jacobian(nterms)
    x = rand(nterms)
    out = similar(x, 1, length(x))
    inp = FastDifferentiation.make_variables(:inp, length(x))


    jac = FastDifferentiation.jacobian([rosenbrock(inp)], inp)
    fd_func = FastDifferentiation.make_function(jac, inp, in_place=true)
    fd_func(x, out)
    @benchmark $fd_func($x, $out)
end
export fd_rosenbrock_jacobian

function fd_rosenbrock_hessian(nterms)
    x = rand(nterms)
    out = fill(eltype(x), (nterms, nterms)) #now have to initialize array before passing it in.
    inp = FastDifferentiation.make_variables(:inp, length(x))


    hess = FastDifferentiation.hessian(rosenbrock(inp), inp)
    println("numberops $(FastDifferentiation.number_of_operations(hess))")
    fd_func = FastDifferentiation.make_function(hess, inp, in_place=true)
    # fd_func(x, out)

    # @benchmark $fd_func($x, $out)
end
export fd_rosenbrock_hessian


function fd_reverse_AD_rosenbrock_jacobian(nterms)
    x = rand(nterms)
    out = similar(x, 1, length(x))
    inp = FastDifferentiation.make_variables(:inp, length(x))


    jac = FastDifferentiation.reverse_AD(rosenbrock(inp), inp)
    fd_func = FastDifferentiation.make_function(jac, inp, in_place=true)

    @benchmark fd_func()
end
export fd_reverse_AD_rosenbrock_jacobian


"""This function pairs with `time_fd_reverse_diff_example`"""
function reverse_diff_R¹⁰⁰R¹⁰⁰(nsize)
    # some objective functions to work with
    f(a, b) = (a + b) * (a * b)'

    # pre-record JacobianTapes for `f` using inputs of shape 10x10 with Float64 elements
    f_tape = ReverseDiff.JacobianTape(f, (rand(nsize, nsize), rand(nsize, nsize)))

    # compile `f_tape` into more optimized representations
    compiled_f_tape = ReverseDiff.compile(f_tape)

    # some inputs and work buffers to play around with
    a, b = rand(nsize, nsize), rand(nsize, nsize)
    inputs = (a, b)

    results = (similar(a, nsize^2, nsize^2), similar(b, nsize^2, nsize^2))

    ####################
    # taking Jacobians #
    ####################

    # with pre-recorded/compiled tapes (generated in the setup above) #
    #-----------------------------------------------------------------#

    # these should be the fastest methods, and non-allocating
    return @benchmark ReverseDiff.jacobian!($results, $compiled_f_tape, $inputs)
end
export reverse_diff_R¹⁰⁰R¹⁰⁰

function reverse_diff_rosenbrock_jacobian(nsize)
end

function reverse_diff_rosenbrock_hessian(nsize)
    input = rand(nsize)
    hcfg = ReverseDiff.HessianConfig(input)
    h_tape = ReverseDiff.HessianTape(rosenbrock, input, hcfg)

    compiled_h_tape = ReverseDiff.compile(h_tape)

    output = rand(nsize, nsize)

    return @benchmark ReverseDiff.hessian!($output, $compiled_h_tape, $input)

end
export reverse_diff_rosenbrock_hessian

function reverse_diff_SHFunctions()
end
export reverse_diff_SHFunctions


"""This FD function is used to compare against both ReverseDiff.jl and Enzyme.jl"""
function fd_R¹⁰⁰R¹⁰⁰(n_size)
    f(a, b) = (a + b) * (a * b)'

    av = FastDifferentiation.make_variables(:ain, n_size^2)
    bv = FastDifferentiation.make_variables(:bin, n_size^2)

    ain = reshape(av, n_size, n_size)
    bin = reshape(bv, n_size, n_size)
    tmp_mat = Matrix{Float64}(undef, n_size^2, length(av) + length(bv))

    orig_func = vec(Node.(f(ain, bin)))
    inputs = vcat(av, bv)
    # @info "Starting symbolic Jacobian"
    dgraph = FastDifferentiation.DerivativeGraph(orig_func)
    # @info "number of operations $(FastDifferentiation.number_of_operations(dgraph))"
    jac = FastDifferentiation.jacobian(orig_func, inputs)
    # @info "Finished symbolic Jcobian, starting make_function"
    fd_func = FastDifferentiation.make_function(jac, inputs, in_place=true)
    # @info "Finshed make_function"


    float_input = rand(2 * n_size^2)
    bench1 = @benchmark $fd_func($float_input, $tmp_mat)
    return bench1
end
export fd_R¹⁰⁰R¹⁰⁰

"""This function pairs with `time_fd_reverse_diff_example`"""
function enzyme_rosenbrock_gradient(nterms)
    x = rand(nterms)
    dx = zeros(nterms)
    # Enzyme.autodiff(Enzyme.Reverse, rosenbrock, Enzyme.Active, Enzyme.Duplicated(x, dx))
    @benchmark Enzyme.autodiff(Enzyme.Reverse, rosenbrock, Enzyme.Active, Enzyme.Duplicated($x, $dx))
    # Enzyme.Compiler.enzyme_code_llvm(stdout, rosenbrock, Enzyme.Active, Tuple{Enzyme.Duplicated{Vector{Float64}}})
end
export enzyme_rosenbrock_gradient

function enzyme_rosenbrock_hessian(nterms)
end

function enzyme_SHFunctions(nterms)
end

function fd_SHFunctions(nterms)
    FastDifferentiation.@variables x y z

    symb_func = SHFunctions(nterms, x, y, z)

    FastDifferentiation.jacobian(symb_func, [x, y, z])

    result = Matrix{Float64}(undef, nterms^2, 3)

    func = FastDifferentiation.make_function(symb_func, SVector(x, y, z), in_place=true)

    @benchmark $func(inputs, $result) setup = inputs = rand(3)
end
export fd_SHFunctions

const hessian_benchmarks = (fd_rosenbrock_hessian, enzyme_rosenbrock_hessian, forward_diff_rosenbrock_hessian, reverse_diff_rosenbrock_hessian)
const jacobian_benchmarks = [
    fd_rosenbrock_jacobian, enzyme_rosenbrock_gradient, forward_diff_rosenbrock_jacobian, reverse_diff_rosenbrock_jacobian,
    fd_R¹⁰⁰R¹⁰⁰, forward_diff_R¹⁰⁰R¹⁰⁰, reverse_diff_R¹⁰⁰R¹⁰⁰]

function run_hessian_benchmarks(nterms)
    benches = [f(nterms) for f in hessian_benchmarks]
    times = [minimum(x.times) for x in benches]

    row = "| "
    for time in times
        row *= " $time |"
    end
    return row
end
export run_hessian_benchmarks

end #module