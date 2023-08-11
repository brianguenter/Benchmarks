function fd_rosenbrock_gradient(nterms)
    x = rand(nterms)
    out = similar(x, 1, length(x))
    inp = FastDifferentiation.make_variables(:inp, length(x))


    jac = FastDifferentiation.jacobian([rosenbrock(inp)], inp)
    fd_func = FastDifferentiation.make_function(jac, inp, in_place=true, init_with_zeros=false)
    fd_func(out, x)
    @benchmark $fd_func($out, $x)
end
export fd_rosenbrock_gradient

function fd_rosenbrock_hessian(nterms)
    x = rand(nterms)
    out = Matrix{eltype(x)}(undef, length(x), length(x))
    inp = FastDifferentiation.make_variables(:inp, length(x))


    hess = FastDifferentiation.hessian(rosenbrock(inp), inp)
    fd_func = FastDifferentiation.make_function(hess, inp, in_place=true)


    @benchmark $fd_func($out, $x)
end
export fd_rosenbrock_hessian


function fd_sparse_placeholder(nterms)
    return ("[^1]", "[^1]: **FD** sparse was slower than **FD** dense so results are only shown for dense.")
end
export fd_sparse_placeholder

function fd_rosenbrock_hessian_sparse(nterms)
    x = rand(nterms)
    inp = FastDifferentiation.make_variables(:inp, length(x))


    hess = FastDifferentiation.sparse_hessian(rosenbrock(inp), inp)

    fd_func = FastDifferentiation.make_function(hess, inp, in_place=true)

    sp_out = similar(hess, Float64)

    @benchmark $fd_func($sp_out, $x)
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


function fd_R¹⁰⁰R¹⁰⁰(n_size)
    f(a, b) = (a + b) * (a * b)'

    av = FastDifferentiation.make_variables(:ain, n_size^2)
    bv = FastDifferentiation.make_variables(:bin, n_size^2)

    ain = reshape(av, n_size, n_size)
    bin = reshape(bv, n_size, n_size)
    tmp_mat = Matrix{Float64}(undef, n_size^2, length(av) + length(bv))

    orig_func = vec(FastDifferentiation.Node.(f(ain, bin)))
    # println("orig ops $(FastDifferentiation.number_of_operations(orig_func))")
    inputs = vcat(av, bv)

    jac = FastDifferentiation.jacobian(orig_func, inputs)
    # println("num ops $(FastDifferentiation.number_of_operations(jac))")
    # println("R100 sparsness $(FastDifferentiation.sparsity(jac))")

    fd_func = FastDifferentiation.make_function(jac, inputs, in_place=true)



    float_input = rand(2 * n_size^2)
    bench1 = @benchmark $fd_func($tmp_mat, $float_input)
    return bench1
end
export fd_R¹⁰⁰R¹⁰⁰


function fd_SHFunctions(nterms)
    FastDifferentiation.@variables x y z

    symb_func = SHFunctions(nterms, x, y, z)

    jac = FastDifferentiation.jacobian(symb_func, [x, y, z])

    result = Matrix{Float64}(undef, nterms^2, 3)

    func = FastDifferentiation.make_function(jac, SVector(x, y, z), in_place=true)

    @benchmark $func($result, inputs) setup = inputs = rand(3)
end
export fd_SHFunctions


function fd_ODE()
    #generate function to be differentiated and store in vector f_y
    y = FastDifferentiation.make_variables(:y, 20)
    f_y = Vector{FastDifferentiation.Node}(undef, 20)
    ODE.f(f_y, y, nothing, nothing)


    ### FD code to compute symbolic Jacobian and make executable

    #compute symbolic derivative
    jac = FastDifferentiation.jacobian(f_y, y)

    println("number of operations $(FastDifferentiation.number_of_operations(jac))")
    #compile to executable
    fd_exe = FastDifferentiation.make_function(jac, y, in_place=true, init_with_zeros=true)

    ### End FD code


    #make input vector and matrix to hold result
    float_J1 = Matrix{Float64}(undef, 20, 20)
    float_y = rand(20)

    return @benchmark $fd_exe($float_J1, $float_y)
end
export fd_ODE

function fd_ODE_sparse()
    #generate function to be differentiated and store in vector f_y
    y = FastDifferentiation.make_variables(:y, 20)
    f_y = Vector{FastDifferentiation.Node}(undef, 20)
    ODE.f(f_y, y, nothing, nothing)


    ### FD code to compute symbolic Jacobian and make executable

    #compute symbolic derivative
    jac = FastDifferentiation.sparse_jacobian(f_y, y)

    #compile to executable
    fd_exe = FastDifferentiation.make_function(jac, y, in_place=true)

    ### End FD code


    #sanity check to make sure derivative is correct
    float_y = rand(20)
    J = similar(jac, Float64)
    Jh = rand(20, 20)

    fd_exe(J, float_y)
    ODE.fjac(Jh, float_y, nothing, nothing)

    @assert isapprox(Jh, J)

    #run benchmark
    return @benchmark $fd_exe($J, $float_y)
end
export fd_ODE_sparse
