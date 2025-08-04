module ACGreen
# import packages
using LinearAlgebra, Random

# export interfaces
export Mesh, make_mesh, UniformMesh, TangentMesh, Cont, Delta, Mixed
export solve, CtxData
export SpectrumType, Cont, Delta, Mixed
export Solver, BarRat, NAC, MaxEntChi2kink, SSK, SAC, SOM, SPX
export curve_fit, LsqFitResult, PronyApproximation
export fdgradient, ∇L2loss
export bfgs, newton

include("globalset.jl")
include("math/math.jl")
include("mesh.jl")
include("solve.jl")
include("barrat.jl")
include("model.jl")
include("maxent.jl")
include("ssk.jl")
include("sac.jl")
include("som.jl")
include("spx.jl")
include("nac.jl")

end
