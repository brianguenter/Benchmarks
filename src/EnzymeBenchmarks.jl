#NOTE: you must set number of threads to 1 or Enzyme will generate slower thread safe code.

function enzyme_rosenbrock_gradient(nterms)
    x = rand(nterms)
    dx = zeros(nterms)

    @benchmark Enzyme.autodiff(Enzyme.Reverse, rosenbrock, Enzyme.Active, Enzyme.Duplicated($x, $dx))

end
export enzyme_rosenbrock_gradient

function enzyme_rosenbrock_hessian(nterms)
    x = rand(nterms)
    dx = zeros(nterms)

    function hess(x, dx)
        tmp = Enzyme.autodiff(Enzyme.Reverse, rosenbrock, Enzyme.Active, Enzyme.Active(x))
        println(tmp)
        return Enzyme.autodiff(Enzyme.Reverse, rosenbrock, Enzyme.Active, Enzyme.Duplicated(tmp[1], tmp[2]))
    end

    hess(x, dx)

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
