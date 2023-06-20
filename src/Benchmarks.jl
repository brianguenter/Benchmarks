module Benchmarks
import FastDifferentiation
using FastDifferentiation: make_variables, make_function, jacobian, Node
using ForwardDiff
using BenchmarkTools
import ReverseDiff
using ReverseDiff: JacobianTape, JacobianConfig, compile, jacobian!
using LinearAlgebra
import Enzyme

using ForwardDiff: GradientConfig, Chunk, gradient!

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
function test_rosenbrock_forward_diff(nterms)
    x = rand(nterms)

    # output buffer
    out = similar(x)
    fastest = typemax(Float64)
    local best_trial

    for chunk_size in 1:3:min(nterms, 10)
        cfg = GradientConfig(rosenbrock, x, Chunk{chunk_size}())
        trial = @benchmark gradient!($out, $rosenbrock, $x, $cfg)
        if minimum(trial).time < fastest
            best_trial = trial
            fastest = minimum(trial).time
        end
    end
    return best_trial
end
export test_rosenbrock_forward_diff

"""This function pairs with `test_rosenbrock_forward_dif`"""
function time_fd_rosenbrock(nterms)
    x = rand(nterms)
    out = similar(x, 1, length(x))
    inp = make_variables(:inp, length(x))


    jac = jacobian([rosenbrock(inp)], inp)
    fd_func = make_function(jac, inp, in_place=true)
    fd_func(x, out)
    @benchmark $fd_func($x, $out)
end
export time_fd_rosenbrock

"""This function pairs with `time_fd_reverse_diff_example`"""
function time_reverse_diff(nsize)
    # some objective functions to work with
    f(a, b) = (a + b) * (a * b)'

    # pre-record JacobianTapes for `f` using inputs of shape 10x10 with Float64 elements
    f_tape = JacobianTape(f, (rand(nsize, nsize), rand(nsize, nsize)))

    # compile `f_tape` into more optimized representations
    compiled_f_tape = compile(f_tape)

    # some inputs and work buffers to play around with
    a, b = rand(nsize, nsize), rand(nsize, nsize)
    inputs = (a, b)
    output = rand(nsize, nsize)
    results = (similar(a, nsize^2, nsize^2), similar(b, nsize^2, nsize^2))
    fcfg = JacobianConfig(inputs)

    ####################
    # taking Jacobians #
    ####################

    # with pre-recorded/compiled tapes (generated in the setup above) #
    #-----------------------------------------------------------------#

    # these should be the fastest methods, and non-allocating
    return @benchmark jacobian!($results, $compiled_f_tape, $inputs)
end
export time_reverse_diff

"""This FD function is used to compare against both ReverseDiff.jl and Enzyme.jl"""
function time_fd_reverse_diff_example(n_size)
    f(a, b) = (a + b) * (a * b)'

    av = make_variables(:ain, n_size^2)
    bv = make_variables(:bin, n_size^2)

    ain = reshape(av, n_size, n_size)
    bin = reshape(bv, n_size, n_size)
    tmp_mat = Matrix{Float64}(undef, n_size^2, length(av) + length(bv))

    orig_func = vec(Node.(f(ain, bin)))
    inputs = vcat(av, bv)
    # @info "Starting symbolic Jacobian"
    dgraph = FastDifferentiation.DerivativeGraph(orig_func)
    # @info "number of operations $(FastDifferentiation.number_of_operations(dgraph))"
    jac = jacobian(orig_func, inputs)
    # @info "Finished symbolic Jcobian, starting make_function"
    fd_func = make_function(jac, inputs, in_place=true)
    # @info "Finshed make_function"

    println("")
    float_input = rand(2 * n_size^2)
    bench1 = @benchmark $fd_func($float_input, $tmp_mat)
    return bench1
end
export time_fd_reverse_diff_example

"""This function pairs with `time_fd_reverse_diff_example`"""
function time_enzyme(nterms)
    x = rand(nterms)
    dx = zeros(nterms)
    @benchmark Enzyme.autodiff(Enzyme.Reverse, rosenbrock, Enzyme.Active, Enzyme.Duplicated($x, $dx))
end
export time_enzyme

end #module