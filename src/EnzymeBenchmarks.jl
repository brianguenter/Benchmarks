#NOTE: you must set number of threads to 1 or Enzyme will generate slower thread safe code.

"""This function pairs with `time_fd_reverse_diff_example`"""
function enzyme_rosenbrock_gradient(nterms)
    x = rand(nterms)
    dx = zeros(nterms)

    @benchmark Enzyme.autodiff(Enzyme.Reverse, rosenbrock, Enzyme.Active, Enzyme.Duplicated($x, $dx))

end
export enzyme_rosenbrock_gradient

function enzyme_rosenbrock_hessian(nterms)
end
export enzyme_rosenbrock_hessian

function enzyme_SHFunctions(nterms)
end
export enzyme_SHFunctions

function enzyme_R¹⁰⁰R¹⁰⁰(nterms)
end
export enzyme_R¹⁰⁰R¹⁰⁰

function enzyme_ODE()
end
export enzyme_ODE
