using ACGreen, LinearAlgebra
using FFTW, Convex, SCS, Plots
include("../../test/testsetup.jl")

noise = 0.0
A, ctx, GFV = dfcfg(Float64, Cont(); npole=2, ml=1000, noise=noise)

N = ctx.N
M = length(ctx.mesh)
F = zeros(M, M)
iden = Matrix{Float64}(I, M, M)
for i in 1:M
    F[:, i] = idct(iden[:, i])
end

K = [ctx.mesh_weight[j] / (ctx.iwn[i] - ctx.mesh[j]) for i in 1:N, j in 1:M]
D = F'*(real(K)'*real(K) + imag(K)'*imag(K))*F
b = F'*(real(K)'*real(GFV) + imag(K)'*imag(GFV))

vx = Variable(M)
problem = minimize(norm(vx, 1), K * F * vx == GFV)
result = solve!(problem, SCS.Optimizer)

reAw = evaluate(vx)

reA1 = idct(reAw)

norm(K*reA1 - GFV)
# 3.487535927477948e-5, noise = 0
# 2.7415356936027946e-5, noise = 1e-6

mesh, reA2 = solve(GFV, ctx, BarRat())

norm(K*reA2 - GFV)
# 8.022398044872697e-7, noise = 0
# 0.0001373023177879882, noise = 1e-6

plot(ctx.mesh, A.(ctx.mesh); xlabel="ω", ylabel="A(ω)", label="original A(ω)",
     title="noise = $noise")
plot!(ctx.mesh, reA1; label="ConpressedSensing")
plot!(ctx.mesh, reA2; label="BarRat")
