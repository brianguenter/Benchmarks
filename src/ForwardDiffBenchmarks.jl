function forward_diff_rosenbrock_gradient(nterms)
    x = rand(nterms)

    # output buffer
    out = similar(x)
    fastest = typemax(Float64)
    local best_trial

    for chunk_size in (1, 4:4:min(nterms, 10)...)
        cfg = ForwardDiff.GradientConfig(rosenbrock, x, ForwardDiff.Chunk{chunk_size}())
        trial = @benchmark ForwardDiff.gradient!($out, $rosenbrock, $x, $cfg)
        if minimum(trial).time < fastest
            best_trial = trial
            fastest = minimum(trial).time
        end
    end
    return best_trial
end
export forward_diff_rosenbrock_gradient

function forward_diff_R¹⁰⁰R¹⁰⁰(nsize)
    function f(x)
        a = view(x, :, 1:nsize)
        b = view(x, :, nsize+1:2*nsize)

        return (a + b) * (a * b)'
    end

    # some inputs and work buffers to play around with

    inputs = rand(nsize, 2 * nsize)

    results = similar(inputs, nsize^2, 2 * nsize^2)

    fastest = typemax(Float64)
    local best_trial

    for chunk_size in 1:3:min(nsize, 20)
        cfg = ForwardDiff.JacobianConfig(f, inputs, ForwardDiff.Chunk{chunk_size}())
        trial = @benchmark ForwardDiff.jacobian!($results, $f, $inputs, $cfg)
        if minimum(trial).time < fastest
            best_trial = trial
            fastest = minimum(trial).time
        end
    end
    return best_trial
end
export forward_diff_R¹⁰⁰R¹⁰⁰

function forward_diff_rosenbrock_hessian(nterms)
    x = rand(nterms)

    # output buffer
    out = similar(x, eltype(x), (length(x), length(x)))
    fastest = typemax(Float64)
    local best_trial
    local best_chunk_size = -1

    for chunk_size in 1:3:min(nterms, 20)
        cfg = ForwardDiff.HessianConfig(rosenbrock, x, ForwardDiff.Chunk{chunk_size}())
        trial = @benchmark ForwardDiff.hessian!($out, $rosenbrock, $x, $cfg)
        if minimum(trial).time < fastest
            best_trial = trial
            fastest = minimum(trial).time
            best_chunk_size = chunk_size
        end
    end
    @info "best chunk size $best_chunk_size"
    return best_trial
end
export forward_diff_rosenbrock_hessian

function forward_diff_SHFunctions(nterms)
    x = rand(3)

    out = similar(x, (nterms^2, 3))
    fastest = typemax(Float64)
    local best_trial
    sh_wrapper(x) = SHFunctions(nterms, x[1], x[2], x[3])

    for chunk_size in 1:3
        cfg = ForwardDiff.JacobianConfig(sh_wrapper, x, ForwardDiff.Chunk{chunk_size}())
        trial = @benchmark ForwardDiff.jacobian($sh_wrapper, $x, $cfg)
        if minimum(trial).time < fastest
            best_trial = trial
            fastest = minimum(trial).time
        end
    end
    return best_trial
end
export forward_diff_SHFunctions

function forward_diff_ODE()
    def_args!(dy, y) = ODE.f(dy, y, nothing, nothing)


    y = copy(ODE.u0)
    dy = Vector{Float64}(undef, 20)
    fastest = typemax(Float64)
    local best_trial

    J_forward = similar(y, 20, 20)
    J_hand = similar(y, 20, 20)

    for chunk_size in 1:3:20
        cfg = ForwardDiff.JacobianConfig(def_args!, dy, y, ForwardDiff.Chunk{chunk_size}())
        trial = @benchmark ForwardDiff.jacobian!($J_forward, $def_args!, $dy, $y, $cfg)

        if minimum(trial).time < fastest
            best_trial = trial
            fastest = minimum(trial).time
        end
    end
    ODE.fjac(J_hand, y, nothing, nothing)
    J_hand ≈ J_forward || error("Inaccurate Jacobian")

    return best_trial
end
export forward_diff_ODE
