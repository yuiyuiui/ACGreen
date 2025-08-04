struct PreComput{T<:Real}
    ss::SingularSpace{T}
    model::Vector{T}
    αvec::Vector{T}
    σ::T
    DS::Diagonal{T,Vector{T}}
    KDw::Matrix{T}
    DSUadDivσ²::Matrix{T}
    S²VadDwDivσ²::Matrix{T}
end
function PreComput(GFV::Vector{Complex{T}}, ctx::CtxData{T},
                   alg::MaxEntChi2kink) where {T<:Real}
    L = alg.L
    α₁ = T(alg.α₁)
    σ = T(ctx.σ)
    w = ctx.mesh_weight
    ss = SingularSpace(GFV, ctx.iwn, ctx.mesh)
    reA = make_model(alg.model_type, ctx)
    αvec = Vector{T}(undef, L)
    αvec[1] = α₁
    for i in 2:L
        αvec[i] = αvec[i-1] / 10
    end
    _, K, _, U, S, V = ss
    KDw = K * Diagonal(w)
    DS = Diagonal(S)
    DSUadDivσ² = DS*U'/σ^2 # to construct J = -V'∂Q/∂A
    S²VadDwDivσ² = DS^2 * V' * Diagonal(w)/σ^2 # to construct H = -V'∂²Q/∂A∂u = ∂J/∂u
    return PreComput(ss, reA, αvec, σ, DS, KDw, DSUadDivσ², S²VadDwDivσ²)
end
function solve(GFV::Vector{Complex{T}}, ctx::CtxData{T},
               alg::MaxEntChi2kink) where {T<:Real}
    ctx.spt isa Cont && alg.maxiter > 1 &&
        error("maxiter>1 is not stable for cont spectrum solve")
    maxiter = alg.maxiter
    pc = PreComput(GFV, ctx, alg)
    reA = pc.model
    for i in 1:maxiter
        pc.model .= reA
        reA = chi2kink(pc)
    end
    if ctx.spt isa Cont
        return ctx.mesh, reA
    elseif ctx.spt isa Delta
        p = ctx.mesh[find_peaks(ctx.mesh, reA, ctx.fp_mp; wind=ctx.fp_ww)]
        γ = pG2γ(p, GFV, ctx.iwn)
        return ctx.mesh, reA, (p, γ)
    else
        error("Unsupported spectral function type")
    end
end

struct MaxEnt_A{T<:Real} <: Function
    model::Vector{T}
    V::Matrix{T}
end
(f::MaxEnt_A{T})(u::Vector{T}) where {T<:Real} = f.model .* exp.(f.V * u)
struct MaxEnt_χ²{T<:Real} <: Function
    G::Vector{T}
    KDw::Matrix{T}
    model::Vector{T}
    V::Matrix{T}
    σ::T
end
function (f::MaxEnt_χ²{T})(u::Vector{T}) where {T<:Real}
    res = (f.G - f.KDw * (f.model .* exp.(f.V * u))) / (f.σ)
    return res' * res
end
struct MaxEnt_J{T<:Real} <: Function
    α::T
    DSUadDivσ²::Matrix{T}
    KDw::Matrix{T}
    model::Vector{T}
    V::Matrix{T}
    G::Vector{T}
end
function (f::MaxEnt_J{T})(u::Vector{T}) where {T<:Real}
    return f.α * u + f.DSUadDivσ² * (f.KDw * (f.model .* exp.(f.V * u)) - f.G)
end
struct MaxEnt_H{T<:Real} <: Function
    α::T
    S²VadDwDivσ²::Matrix{T}
    model::Vector{T}
    V::Matrix{T}
end
function (f::MaxEnt_H{T})(u::Vector{T}) where {T<:Real}
    return f.α * I(size(f.V, 2)) + f.S²VadDwDivσ² * Diagonal(f.model .* exp.(f.V * u)) * f.V
end

function G2χ²vec(pc::PreComput{T}) where {T<:Real}
    G, _, n, _, _, V = pc.ss
    αvec = pc.αvec
    χ²vec = Vector{T}(undef, length(αvec))
    _χ² = MaxEnt_χ²(G, pc.KDw, pc.model, V, pc.σ)
    # Now solve the minimal with Newton method
    u_guess = zeros(T, n)
    u_opt_vec = Vector{Vector{T}}(undef, length(αvec))
    for i in 1:length(αvec)
        #@show i
        α = αvec[i]
        u_opt, call, _ = newton(MaxEnt_J(α, pc.DSUadDivσ², pc.KDw, pc.model, V, G),
                                MaxEnt_H(α, pc.S²VadDwDivσ², pc.model, V), u_guess)
        u_guess = copy(u_opt)
        u_opt_vec[i] = copy(u_opt)
        χ²vec[i] = _χ²(u_opt)
        #@show log10(α), log10(χ²_vec[i]), norm(J(u_opt, α)), call
    end
    idx = findall(isfinite, χ²vec)
    return u_opt_vec, χ²vec, idx
end
function χ²vec2αopt(χ²vec::Vector{T}, αvec::Vector{T}) where {T<:Real}
    # Now performe curve fit
    guess_fit = [T(0), T(5), T(2), T(0)]
    function fitfun(x, p)
        return @. p[1] + p[2] / (T(1) + exp(-p[4] * (x - p[3])))
    end
    p = curve_fit(fitfun, log10.(αvec), log10.(χ²vec), guess_fit).param
    # choose the inflection point as the best α
    # Parameter to prevent overfitting when fitting the curve
    adjust = T(5//2)
    αopt = 10^(p[3]-adjust/p[4])
    return αopt
end
function chi2kink(pc::PreComput{T}) where {T<:Real}
    _A = MaxEnt_A(pc.model, pc.ss.V)
    u_opt_vec, χ²vec, idx = G2χ²vec(pc)
    αopt = χ²vec2αopt(χ²vec[idx], pc.αvec[idx])
    u_guess = copy(u_opt_vec[findmin(abs.(log10.(pc.αvec) .- log10(αopt)))[2]])
    u_opt, = newton(MaxEnt_J(αopt, pc.DSUadDivσ², pc.KDw, pc.model, pc.ss.V, pc.ss.G),
                    MaxEnt_H(αopt, pc.S²VadDwDivσ², pc.model, pc.ss.V), u_guess)
    # recover the A
    return _A(u_opt)
end
