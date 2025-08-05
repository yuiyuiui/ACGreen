function dct(x::AbstractVector{T}) where {T<:Real}
    N = length(x)
    y = Vector{T}(undef, N)
    invN = T(1/N)
    sqrt1N = sqrt(invN)       # α₀ = √(1/N)
    sqrt2N = sqrt(2*invN)     # αₖ = √(2/N) for k>0

    for k in 0:(N - 1)
        s = T(0)
        θk = π * k / N
        for n in 0:(N - 1)
            s += x[n+1] * cos(θk * (n + T(1//2)))
        end
        α = (k == 0 ? sqrt1N : sqrt2N)
        y[k+1] = α * s
    end

    return y
end

function idct(X::Vector{T}) where {T<:Real}
    N = length(X)
    x = similar(X, T)

    Xmod = copy(X)
    Xmod[1] *= T(1 / sqrt(2))

    factor = sqrt(T(2 / N))

    for n in 0:(N - 1)
        x[n+1] = factor * sum(Xmod[k+1] * cos(π / N * k * (n + T(1//2))) for k in 0:(N - 1))
    end

    return x
end
