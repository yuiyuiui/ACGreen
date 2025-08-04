function slicerightmul!(A::Array{T,3}, B::Matrix{T}, p::Int) where {T}
    A[:, :, p] .= A[:, :, p] * B
    return A
end

function sliceleftmul!(A::Array{T,3}, B::Matrix{T}, p::Int) where {T}
    A[:, :, p] .= B * A[:, :, p]
    return A
end
