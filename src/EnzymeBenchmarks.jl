#NOTE: you must set number of threads to 1 or Enzyme will generate slower thread safe code.

function enzyme_rosenbrock_gradient(nterms)
    x = rand(nterms)
    dx = zeros(nterms)

    @benchmark Enzyme.autodiff(Enzyme.Reverse, rosenbrock, Enzyme.Active, Enzyme.Duplicated($x, $dx))

end
export enzyme_rosenbrock_gradient

"""example from Enzyme documentation. Doesn't work"""
function test_enzyme()
    function f(x::Array{Float64}, y::Array{Float64})
        y[1] = x[1] * x[1] * x[1] + x[2] * x[2] * x[2]
        return nothing
    end

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
    return vdbx
end
export test_enzyme


function enzyme_rosenbrock_hessian(nterms)
    function f(x::Array{Float64}, y::Array{Float64})
        y[1] = rosenbrock(x)
    end

    y = [0.0]
    x = rand(nterms)

    vdy = ntuple(i -> [0.0], nterms)
    vdx = ntuple(i -> begin
            tmp = zeros(nterms)
            tmp[i] = 1
            return tmp
        end, nterms)

    bx = zeros(nterms)
    by = [1.0]
    vdbx = ntuple(i -> zeros(nterms), nterms)

    vdby = ntuple(i -> [0.0], nterms)

    return @benchmark Enzyme.autodiff(
        Enzyme.Forward,
        (x, y) -> Enzyme.autodiff_deferred(Enzyme.Reverse, $f, x, y),
        Enzyme.BatchDuplicated(Enzyme.Duplicated($x, $bx), Enzyme.Duplicated.($vdx, $vdbx)),
        Enzyme.BatchDuplicated(Enzyme.Duplicated($y, $by), Enzyme.Duplicated.($vdy, $vdby)),
    )
end
export enzyme_rosenbrock_hessian

# code for enzyme rosenbrock hessian. Uses @generated so some code has to be at top level

# NTERMS = 100
# export NTERMS

# x__ = SVector{NTERMS,Float64}(rand(NTERMS))
# undefvar__ = MVector{NTERMS,Float64}(rand(NTERMS))
# @generated function onehott()
#     res = []
#     for i in 1:NTERMS
#         tmp = []
#         for j in 1:NTERMS
#             if i == j
#                 push!(tmp, 1.0)
#             else
#                 push!(tmp, 0.0)
#             end
#         end
#         push!(res, SVector{NTERMS,Float64}(tmp...))
#     end
#     :($((res...,)))
# end
# onehott()

# hess__ = ntuple(Val(NTERMS)) do i
#     Base.@_inline_meta
#     MVector{NTERMS,Float64}(zeros(NTERMS))
# end

# function tmp(x, dx)
#     dx .= Enzyme.autodiff_deferred(Enzyme.Reverse, rosenbrock, Enzyme.Active(x))[1][1]
#     nothing
# end

# bd1__ = Enzyme.BatchDuplicatedFunc{SVector{NTERMS,Float64},NTERMS,typeof(onehott)}(x__)
# bd2__ = Enzyme.BatchDuplicatedNoNeed(undefvar__, hess__)


# function time_hessian()
#     @benchmark Enzyme.autodiff(Enzyme.Forward, $tmp, $bd1__, $bd2__)
# end
# export time_hessian

#end code for enzyme rosenbrock hessian

function enzyme_SHFunctions(nterms)
    return ("[^5.1]", "[^5.1]: Enzyme crashes Julia REPL for SHFunctions benchmark.")

    f(x) = SHFunctions(nterms, x[1], x[2], x[3])

    vin = rand(3)

    #This crashes 
    Enzyme.jacobian(Enzyme.Forward, f, vin, Val(nterms^2))
end
export enzyme_SHFunctions

function enzyme_R¹⁰⁰R¹⁰⁰(nterms)
    return ("[^5]", "[^5]: Enzyme prints \"Warning: using fallback BLAS replacements, performance may be degraded\", followed by stack overflow error or endless loop.")
    function wrapf(v)
        rows, cols = size(v)
        a = view(v, :, 1:(cols÷2))
        b = view(v, :, (cols÷2+1):cols)
        return (a + b) * (a * b)'
    end

    composite_size = (nterms^2, 2 * nterms^2)
    vin = rand(composite_size...)

    #This doesn't work locks up terminal
    Enzyme.jacobian(Enzyme.Reverse, wrapf, vin, Val(prod(composite_size)))

    #This doesn't work locks up terminal
    # Enzyme.jacobian(Enzyme.Forward, wrapf, vin, Val(prod(composite_size)))

end
export enzyme_R¹⁰⁰R¹⁰⁰

function enzyme_ODE()
    nterms = 20
    y = rand(nterms)
    @benchmark Enzyme.jacobian(Enzyme.Forward, ODE.enzyme_f, $y, Val($nterms))
end
export enzyme_ODE

function enzyme_tuple_ODE()

    k1 = .35e0

    k2 = .266e2

    k3 = .123e5

    k4 = .86e-3

    k5 = .82e-3

    k6 = .15e5

    k7 = .13e-3

    k8 = .24e5

    k9 = .165e5

    k10 = .9e4

    k11 = .22e-1

    k12 = .12e5

    k13 = .188e1

    k14 = .163e5

    k15 = .48e7

    k16 = .35e-3

    k17 = .175e-1

    k18 = .1e9

    k19 = .444e12

    k20 = .124e4

    k21 = .21e1

    k22 = .578e1

    k23 = .474e-1

    k24 = .178e4

    k25 = .312e1



    function tuple_f(y)

        r1 = k1 * y[1]

        r2 = k2 * y[2] * y[4]

        r3 = k3 * y[5] * y[2]

        r4 = k4 * y[7]

        r5 = k5 * y[7]

        r6 = k6 * y[7] * y[6]

        r7 = k7 * y[9]

        r8 = k8 * y[9] * y[6]

        r9 = k9 * y[11] * y[2]

        r10 = k10 * y[11] * y[1]

        r11 = k11 * y[13]

        r12 = k12 * y[10] * y[2]

        r13 = k13 * y[14]

        r14 = k14 * y[1] * y[6]

        r15 = k15 * y[3]

        r16 = k16 * y[4]

        r17 = k17 * y[4]

        r18 = k18 * y[16]

        r19 = k19 * y[16]

        r20 = k20 * y[17] * y[6]

        r21 = k21 * y[19]

        r22 = k22 * y[19]

        r23 = k23 * y[1] * y[4]

        r24 = k24 * y[19] * y[1]

        r25 = k25 * y[20]



        return (-r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25, -r2 - r3 - r9 - r12 + r1 + r21, -r15 + r1 + r17 + r19 + r22, -r2 - r16 - r17 - r23 + r15, -r3 + r4 + r4 + r6 + r7 + r13 + r20, -r6 - r8 - r14 - r20 + r3 + r18 + r18, -r4 - r5 - r6 + r13, r4 + r5 + r6 + r7, -r7 - r8, -r12 + r7 + r9, -r9 - r10 + r8 + r11, r9, -r11 + r10, -r13 + r12, r14, -r18 - r19 + r16, -r20, r20, -r21 - r22 - r24 + r23 + r25, -r25 + r24)

    end



    float_y = rand(20)



    tfloat_y = tuple(float_y...)



    makeOneHot() = Enzyme.onehot(NTuple{20,Float64})



    function jac(y)

        mode = Enzyme.Forward

        func = tuple_f

        Enzyme.autodiff(mode, Enzyme.Const(func), Enzyme.DuplicatedNoNeed, Enzyme.BatchDuplicatedFunc{typeof(y),20,typeof(makeOneHot)}(y))

    end




    return @benchmark $jac($tfloat_y)
end
export enzyme_tuple_ODE
