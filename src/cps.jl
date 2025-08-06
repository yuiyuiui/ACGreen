function cps(signal, rate)
    n = length(signal)
    m = round(Int, n * rate)
    idx = randperm(n)[1:m]
    b = signal[idx]
    I_n = Matrix{Float64}(I, n, n)

    A = zeros(n, n)
    for j in 1:n
        A[:, j] = idct(I_n[:, j])
    end
    A = A[idx, :]

    vx = Variable(n)
    problem = minimize(norm(vx, 1), A * vx == b)
    solve!(problem, SCS.Optimizer; silent=false)
    return evaluate(vx)
end

function solve(GFV::Vector{Complex{T}}, ctx::CtxData, alg::CPS) where {T<:Real}
    G, _, _, U, S, V = SingularSpace(GFV, ctx.iwn, ctx.mesh)
    m = length(ctx.mesh)
    vx = Variable(m)
    A = Diagonal(S) * V'
    problem = minimize(alg.λ * norm(V' * vx, 1) + norm(A * vx - U' * G, 2))
    solve!(problem, SCS.Optimizer; silent=false)
    reĀ = evaluate(vx)
    return ctx.mesh, reĀ ./ ctx.mesh_weight
end
