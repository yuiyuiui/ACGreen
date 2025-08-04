tolerance(T) = eps(real(T))^(1 // 2)
strict_tol(T) = eps(real(T))^(2 // 3)
relax_tol(T) = eps(real(T))^(1 // 4)

# generate data
# construct combanition of gauss waves
function continous_spectral_density(μ::Vector{T},
                                    σ::Vector{T},
                                    amplitudes::Vector{T}) where {T<:Real}
    @assert length(μ) == length(σ) == length(amplitudes)
    n = length(μ)
    function y(x::T)
        res = T(0)
        for i in 1:n
            res += amplitudes[i] * exp(-(x - μ[i])^2 / (2 * σ[i]^2))
        end
        return res
    end
    return y
end

# generate values of G(iw_n)
function generate_GFV_cont(β::T,
                           N::Int,
                           A::Function;
                           int_low::T=(-T(20)),
                           int_up::T=T(20),
                           noise::T=T(0),) where {T<:Real}
    grid = (collect(0:(N - 1)) .+ 1 // 2) * T(2π) / β
    n = length(grid)
    res = zeros(Complex{T}, n)
    for i in 1:n
        res[i] = ACGreen.integral(x -> A(x) / (im * grid[i] - x), int_low, int_up)
    end
    for i in 1:n
        res[i] += noise * randn(T) * res[i] * exp(T(2π) * im * rand(T))
    end
    return res
end

function generate_GFV_delta(β::T,
                            N::Int,
                            poles::Vector{T},
                            γ_vec::Vector{T};
                            noise::T=T(0),) where {T<:Real}
    @assert length(poles) == length(γ_vec)
    wn = (collect(0:(N - 1)) .+ 1 // 2) * T(2π) / β
    res = zeros(Complex{T}, N)
    for i in 1:N
        for j in 1:length(poles)
            res[i] += γ_vec[j] / (im * wn[i] - poles[j])
        end
    end
    for i in 1:N
        res[i] += noise * randn(T) * res[i] * exp(T(2π) * im * rand(T))
    end
    return res
end

# loss function
function loss(G::Vector{T}, G₀::Vector{T}, w::Vector{S}) where {T<:Number,S<:Real}
    @assert S == real(T)
    return sqrt(sum(abs2.(G .- G₀) .* w))
end

function χ²(p, G, wn)
    res = 0
    for (Gⱼ, wⱼ) in zip(G, wn)
        res += abs2(sum(1 ./ (im * wⱼ .- p)) / length(p) - Gⱼ)
    end
    return res
end

# default configuration
function dfcfg(T::Type{<:Real}, spt::SpectrumType;
               μ=[T(1 // 2), T(-5 // 2)]::Vector{T},
               σ=[T(1 // 5), T(4 // 5)]::Vector{T},
               amplitudes=[T(1), T(3 // 10)]::Vector{T},
               mesh_type=UniformMesh(),
               β=T(10)::T,
               N=20,
               GFVσ=T(1e-4)::T,
               noise=T(0)::T,
               mb=T(8)::T,
               ml=801::Int,
               npole::Int=2,
               fp_ww::Real=T(0.01),
               fp_mp::Real=T(0.1))
    ctx = CtxData(spt, β, N; mesh_bound=mb, mesh_length=ml, mesh_type=mesh_type, σ=GFVσ,
                  fp_ww=fp_ww, fp_mp=fp_mp)
    if spt isa Cont
        A = continous_spectral_density(μ, σ, amplitudes)
        Asum = sum(A.(ctx.mesh) .* ctx.mesh_weight)
        orA = x -> A(x) / Asum
        GFV = generate_GFV_cont(β, N, orA; noise=noise)
        return orA, ctx, GFV
    elseif spt isa Delta
        poles = collect(1:npole) .+ rand(T, npole) * T(1 // 2)
        γ = ones(T, npole) ./ npole
        GFV = generate_GFV_delta(β, N, poles, γ; noise=noise)
        return (poles, γ), ctx, GFV
    end
end
