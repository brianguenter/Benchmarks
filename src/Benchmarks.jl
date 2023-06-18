module Benchmarks
using FastDifferentiation
using ForwardDiff
using BenchmarkTools

using ForwardDiff: GradientConfig, Chunk, gradient!

# let's use a Rosenbrock function as our target function
function rosenbrock(x)
           a = one(eltype(x))
           b = 100 * a
           result = zero(eltype(x))
           for i in 1:length(x)-1
               result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
           end
           return result
       end

function test_rosenbrock()
    x = rand(10000)

# output buffer
out = similar(x);

# construct GradientConfig with various chunk sizes
 cfg1 = GradientConfig(rosenbrock, x, Chunk{1}())
 cfg4 = GradientConfig(rosenbrock, x, Chunk{4}())
 cfg10 = GradientConfig(rosenbrock, x, Chunk{10}())
result = Vector{Any}(undef,0)

 for chunk_size in 1:3:10
    @time gradient!(out, rosenbrock, x, cfg1)
  0.775139 seconds (4 allocations: 160 bytes)

# (input length of 10000) / (chunk size of 4) = (2500 4-element chunks)
@time gradient!(out, rosenbrock, x, cfg4)
  0.386459 seconds (4 allocations: 160 bytes)

  @time gradient!(out, rosenbrock, x, cfg10)
# (input length of 10000) / (chunk size of 10) = (100
end #module