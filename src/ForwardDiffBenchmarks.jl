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
end

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
