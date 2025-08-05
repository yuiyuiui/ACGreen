using ACGreen, Plots
include("../../test/testsetup.jl")

T = Float64
A, ctx, GFV = dfcfg(T, Cont(); ml=1000)

alg = CPS()

mesh, reA = solve(GFV, ctx, alg)

plot(ctx.mesh, reA; label="cps")
plot!(ctx.mesh, A.(ctx.mesh); label="original")
