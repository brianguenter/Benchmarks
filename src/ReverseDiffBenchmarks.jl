
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

function reverse_diff_rosenbrock_gradient(nsize)
    input = rand(nsize)
    jcfg = ReverseDiff.JacobianConfig(input)
    j_tape = ReverseDiff.JacobianTape(rosenbrock, input, jcfg)

    compiled_h_tape = ReverseDiff.compile(j_tape)

    output = rand(nsize, nsize)

    return @benchmark ReverseDiff.jacobian!($output, $compiled_h_tape, $input)
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

function reverse_diff_SHFunctions(order)
end
export reverse_diff_SHFunctions

function reverse_diff_ODE()

end
export reverse_diff_ODE