function zygote_rosenbrock_gradient(nterms)
    x = rand(nterms)
    @benchmark Zygote.gradient(rosenbrock, $x)
end
export zygote_rosenbrock_gradient

function zygote_rosenbrock_hessian(nterms)
    x = rand(nterms)
    @benchmark Zygote.hessian(rosenbrock, $x)
end
export zygote_rosenbrock_hessian

function zygote_R¹⁰⁰R¹⁰⁰(n_size)
    f(a, b) = (a + b) * (a * b)'
    x1 = rand(n_size, n_size)
    x2 = rand(n_size, n_size)

    @benchmark Zygote.jacobian($f, $x1, $x2)
end
export zygote_R¹⁰⁰R¹⁰⁰

#has problems with memoize maybe?
function zygote_SHFunctions(nterms)
    wrap(x, y, z) = zygote_SHFunctions(nterms, x, y, z)
    x = rand()
    y = rand()
    z = rand()
    Zygote.jacobian(wrap, x, y, z)
end
export zygote_SHFunctions

function zygote_ODE()
    y = rand(20)

    @benchmark Zygote.jacobian(ODE.zygote_f, $y)
end
export zygote_ODE