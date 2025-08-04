@testset "uniform mesh" begin
    ml = 100
    for T in [Float32, Float64]
        mb = T(5)
        mesh, mesh_weight = make_mesh(mb, ml, UniformMesh())
        @test typeof(mesh) === Vector{T}
        @test typeof(mesh_weight) === Vector{T}
        @test length(mesh) == ml
        @test length(mesh_weight) == ml
        @test mesh == collect(range(-mb, mb, ml))
        @test isapprox(sum(mesh_weight), mb*2, atol=tolerance(T))
    end
end

@testset "tangent mesh" begin
    ml = 100
    for T in [Float32, Float64]
        mb = T(5)
        mesh, mesh_weight = make_mesh(mb, ml, TangentMesh())
        @test typeof(mesh) === Vector{T}
        @test typeof(mesh_weight) === Vector{T}
        @test length(mesh) == ml
        @test length(mesh_weight) == ml
        @test mesh ==
              tan.(collect(range(-T(π)/T(2.1), T(π)/T(2.1), ml)))/tan(T(π)/T(2.1))*mb
        @test isapprox(sum(mesh_weight), mb*2, atol=tolerance(T))
    end
end

@testset "SingularSpace" begin
    n = 10
    for T in [Float32, Float64]
        v1 = Vector{T}(1:n)
        v2 = Vector{Complex{T}}(1:n)
        _, ctx, GFV = dfcfg(T, Cont())
        ss = ACGreen.SingularSpace(GFV, ctx.iwn, ctx.mesh)
        G, K, n, U, S, V = ss
        @test typeof(ss) <: ACGreen.SingularSpace{T}
        kernel = Matrix{Complex{T}}(undef, length(GFV), length(ctx.mesh))
        for i in 1:length(GFV)
            for j in 1:length(ctx.mesh)
                kernel[i, j] = 1 / (ctx.iwn[i] - ctx.mesh[j])
            end
        end
        G0 = vcat(real(GFV), imag(GFV))
        K0 = [real(kernel); imag(kernel)]
        @test G == G0
        @test isapprox(K, K0, atol=n*strict_tol(T))
    end
end

@testset "nearest" begin
    N = 1000000
    for T in [Float32, Float64]
        v = collect(range(T(0), T(1), N))
        r = rand(T)
        idx = findmin(abs.(v .- r))[2]
        @test idx == ACGreen.nearest(v, r)
    end
end
