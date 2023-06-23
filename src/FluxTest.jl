# using Flux

# function main()
#     model = Chain(
#         Dense(2 => 20, tanh),
#         Dense(20 => 2))

#     θ, restructure = Flux.destructure(model)

#     x = make_variables(:x, 2)
#     Zygote.gradient(θ) do θ′
#         model′ = restructure(θ′)
#         _, J = jacobian(x) do x′
#             model′(x′)
#         end
#         tr(J) + sum(J .^ 2) / 2
#     end
# end