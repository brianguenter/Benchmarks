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
    out = Matrix{eltype(x)}(undef, length(x), length(x))
    inp = FastDifferentiation.make_variables(:inp, length(x))


    hess = FastDifferentiation.hessian(rosenbrock(inp), inp)
    fd_func = FastDifferentiation.make_function(hess, inp, in_place=true)


    @benchmark $fd_func($x, $out)
end
export fd_rosenbrock_hessian

"""hesian of rosenbrock is incredibly sparse. Most time in fd exe is spent setting array elements to zero"""
function fd_rosenbrock_hessian_sparse(nterms)
    x = rand(nterms)
    inp = FastDifferentiation.make_variables(:inp, length(x))


    hess = FastDifferentiation.sparse_hessian(rosenbrock(inp), inp)

    fd_func = FastDifferentiation.make_function(hess, inp, in_place=true)

    sparse_array = similar(hess, Float64)
    sp_out = sparse_array(x)

    @benchmark $fd_func($x, $sp_out)
end
export fd_rosenbrock_hessian_sparse

function fd_reverse_AD_rosenbrock_jacobian(nterms)
    x = rand(nterms)
    out = similar(x, 1, length(x))
    inp = FastDifferentiation.make_variables(:inp, length(x))


    jac = FastDifferentiation.reverse_AD(rosenbrock(inp), inp)
    fd_func = FastDifferentiation.make_function(jac, inp, in_place=true)

    @benchmark fd_func()
end
export fd_reverse_AD_rosenbrock_jacobian

"""This FD function is used to compare against both ReverseDiff.jl and Enzyme.jl"""
function fd_R¹⁰⁰R¹⁰⁰(ignore)
    global n_size
    f(a, b) = (a + b) * (a * b)'

    av = FastDifferentiation.make_variables(:ain, n_size^2)
    bv = FastDifferentiation.make_variables(:bin, n_size^2)

    ain = reshape(av, n_size, n_size)
    bin = reshape(bv, n_size, n_size)
    tmp_mat = Matrix{Float64}(undef, n_size^2, length(av) + length(bv))

    orig_func = vec(FastDifferentiation.Node.(f(ain, bin)))
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