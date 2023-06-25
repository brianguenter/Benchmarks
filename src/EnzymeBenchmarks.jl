

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