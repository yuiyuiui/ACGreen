using ACGreen, LinearAlgebra, Random, Plots, Lasso
include("../../test/testsetup.jl")

T = Float64
ml = 1000
fb = 100
orA, ctx, GFV = dfcfg(T, Cont(); ml=ml)

G = vcat(real(GFV), imag(GFV))

K = ACGreen.make_kernel(ctx.mesh, ctx.wn)

K *= Diagonal(ctx.mesh_weight)

K = [real(K); imag(K)]

iden = Matrix{T}(I, ml, ml)

Finv = zeros(T, ml, fb)

for i in 1:fb
    Finv[:, i] = idct(iden[:, i])
end

B = K * Finv

path = fit(LassoPath, B, G;
           λminratio=1e-6,
           maxncoef=size(B, 2))

reAw = collect(path.coefs[:, 63])
norm(B * reAw - G)

plot(ctx.mesh, Finv * reAw)
