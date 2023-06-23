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


using ForwardDiff: GradientConfig, Chunk, gradient!

# include("ODE.jl")
include("SphericalHarmonics.jl")

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
end

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
export fd_SHFunctions_jacobian

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