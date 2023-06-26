

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

function enzyme_R¹⁰⁰R¹⁰⁰()
end

function enzyme_ODE()
end
