function make_kernel(mesh::Vector{T}, grid::Vector{T};
                     grid_type::String="mats_freq",
                     β::Union{T,Missing}=missing) where {T<:Real}
    nmesh = length(mesh)
    ngrid = length(grid)
    if grid_type == "mats_freq"
        kernel = zeros(Complex{T}, nmesh, ngrid)
        for i in 1:nmesh
            for j in 1:ngrid
                kernel[j, i] = 1 / (im * grid[j] - mesh[i])
            end
        end
    elseif grid_type == "imag_time"
        kernel = zeros(T, nmesh, ngrid)
        for i in 1:nmesh
            for j in 1:ngrid
                kernel[j, i] = exp(-grid[j] * mesh[i]) / (1 + exp(-β * mesh[i]))
            end
        end
    end
    return kernel
end
