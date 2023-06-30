#NOTE: you must set number of threads to 1 or Enzyme will generate slower thread safe code.

function enzyme_rosenbrock_gradient(nterms)
    x = rand(nterms)
    dx = zeros(nterms)

    @benchmark Enzyme.autodiff(Enzyme.Reverse, rosenbrock, Enzyme.Active, Enzyme.Duplicated($x, $dx))

end
export enzyme_rosenbrock_gradient

"""example from Enzyme documentation. Doesn't work"""
function test_enzyme()
    f(x) = x[1]^2 * x[2]^2
    y = [0.0]
    x = [2.0, 2.0]

    vdy = ([0.0], [0.0])
    vdx = ([1.0, 0.0], [0.0, 1.0])

    bx = [0.0, 0.0]
    by = [1.0]
    vdbx = ([0.0, 0.0], [0.0, 0.0])
    vdby = ([0.0], [0.0])

    Enzyme.autodiff(
        Enzyme.Forward,
        (x, y) -> Enzyme.autodiff_deferred(Enzyme.Reverse, f, x, y),
        Enzyme.BatchDuplicated(Enzyme.Duplicated(x, bx), Enzyme.Duplicated.(vdx, vdbx)),
        Enzyme.BatchDuplicated(Enzyme.Duplicated(y, by), Enzyme.Duplicated.(vdy, vdby)),
    )
end
export test_enzyme


function enzyme_rosenbrock_hessian(nterms)
    y = [0.0]
    x = rand(nterms)

    vdy = NTuple{nterms,Vector{Float64}}(([0.0] for _ in 1:nterms))
    vdx = Vector{Vector{Float64}}(undef, nterms)
    for i in 1:nterms
        tmp = zeros(nterms)
        tmp[i] = 1.0
        vdx[i] = tmp
    end

    bx = zeros(nterms)
    by = [1.0]
    vdbx = Vector{Vector{Float64}}(undef, nterms)
    for i in 1:nterms
        tmp = zeros(nterms)
        tmp[i] = 1.0
        vdbx[i] = tmp
    end
    vdby = NTuple{nterms,Vector{Float64}}(([0.0] for _ in 1:nterms))

    # Enzyme.Duplicated(x, bx)
    # Enzyme.Duplicated.(vdx, vdbx)
    # Enzyme.Duplicated(y, by)
    # Enzyme.Duplicated.(vdy, vdby)

    Enzyme.autodiff(
        Enzyme.Forward,
        (x, y) -> Enzyme.autodiff_deferred(Reverse, rosenbrock, x, y),
        Enzyme.BatchDuplicated(Enzyme.Duplicated(x, bx), Enzyme.Duplicated.(vdx, vdbx)),
        Enzyme.BatchDuplicated(Enzyme.Duplicated(y, by), Enzyme.Duplicated.(vdy, vdby)),
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

    composite_size = (nterms^2, 2 * nterms^2)
    vin = rand(composite_size...)

    #This doesn't work locks up terminal
    Enzyme.jacobian(Reverse, wrapf, vin, Val(prod(composite_size)))

    #This doesn't work locks up terminal
    Enzyme.jacobian(Forward, wrapf, vin, Val(prod(composite_size)))

end
export enzyme_R¹⁰⁰R¹⁰⁰

function enzyme_ODE()
end
export enzyme_ODE
