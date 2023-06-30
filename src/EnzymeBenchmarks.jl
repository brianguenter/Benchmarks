#NOTE: you must set number of threads to 1 or Enzyme will generate slower thread safe code.

function enzyme_rosenbrock_gradient(nterms)
    x = rand(nterms)
    dx = zeros(nterms)

    @benchmark Enzyme.autodiff(Enzyme.Reverse, rosenbrock, Enzyme.Active, Enzyme.Duplicated($x, $dx))

end
export enzyme_rosenbrock_gradient

function enzyme_rosenbrock_hessian(nterms)
    y = [0.0]
    x = rand(nterms)

    vdy = NTuple{nterms,Float64}(([0.0] for _ in 1:nterms))
    vdx = ([1.0, 0.0], [0.0, 1.0])

    bx = NTuple{nterms,Float64}(([0.0] for _ in 1:nterms))
    by = [1.0]
    vdbx = ([0.0, 0.0], [0.0, 0.0])
    vdby = ([0.0], [0.0])

    Enzyme.autodiff(
        Forward,
        (x, y) -> Enzyme.autodiff_deferred(Reverse, rosenbrock, x, y),
        BatchDuplicated(Duplicated(x, bx), Duplicated.(vdx, vdbx)),
        BatchDuplicated(Duplicated(y, by), Duplicated.(vdy, vdby)),
    )
end
export enzyme_rosenbrock_hessian

function enzyme_SHFunctions(nterms)
    return nothing #until this works

    f(x) = SHFunctions(nterms, x[1], x[2], x[3])

    vin = rand(3)

    #This crashes 
    Enzyme.jacobian(Enzyme.Forward, f, vin, Val(nterms^2))
end
export enzyme_SHFunctions

function enzyme_R¹⁰⁰R¹⁰⁰(nterms)
    return nothing #until this works
    function wrapf(v)
        rows, cols = size(v)
        a = view(v, :, 1:(cols÷2))
        b = view(v, :, (cols÷2+1):cols)
        return (a + b) * (a * b)'
    end

    vin = rand(10, 20)

    #This doesn't work locks up terminal
    Enzyme.jacobian(Reverse, wrapf, vin, Val(100))

    #This doesn't work locks up terminal
    Enzyme.jacobian(Forward, wrapf, vin, Val(100))

end
export enzyme_R¹⁰⁰R¹⁰⁰

function enzyme_ODE()
end
export enzyme_ODE
