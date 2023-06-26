"""This function pairs with `time_fd_rosenbrock`"""
function forward_diff_rosenbrock_jacobian(nterms)
    x = rand(nterms)

    # output buffer
    out = similar(x)
    fastest = typemax(Float64)
    local best_trial

    for chunk_size in 1:3:min(nterms, 10)
        cfg = ForwardDiff.GradientConfig(rosenbrock, x, ForwardDiff.Chunk{chunk_size}())
        trial = @benchmark ForwardDiff.gradient!($out, $rosenbrock, $x, $cfg)
        if minimum(trial).time < fastest
            best_trial = trial
            fastest = minimum(trial).time
        end
    end
    return best_trial
end
export forward_diff_rosenbrock_jacobian

function forward_diff_R¹⁰⁰R¹⁰⁰(nsize)
    # some objective functions to work with
    f(a, b) = (a + b) * (a * b)'

    # some inputs and work buffers to play around with
    a, b = rand(nsize, nsize), rand(nsize, nsize)
    inputs = [a, b]

    results = (similar(a, nsize^2, nsize^2), similar(b, nsize^2, nsize^2))

    fastest = typemax(Float64)
    local best_trial

    for chunk_size in 1:3:min(nsize, 10)
        cfg = ForwardDiff.GradientConfig(f, inputs, ForwardDiff.Chunk{chunk_size}())
        trial = @benchmark ForwardDiff.gradient!($results, $f, $x, $cfg)
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

    for chunk_size in 1:1:3
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
    swap_args!(y, x) = ODE.f(x, y, nothing, nothing)

    y = rand(20)
    dy = Vector{Float64}(undef, 20)

    for chunk_size in 1:3:20
        cfg = ForwardDiff.JacobianConfig(swap_args!, y, dy, ForwardDiff.Chunk{chunk_size}())
        trial = @benchmark ForwardDiff.jacobian($swap_args!, $y, $cfg)
        if minimum(trial).time < fastest
            best_trial = trial
            fastest = minimum(trial).time
        end
    end

    return best_trial
end
export forward_diff_ODE

