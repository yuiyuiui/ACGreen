# CPS项目集成C++库示例
# 基于你的cps.jl文件，展示如何用C++库加速计算

using ACGreen, LinearAlgebra
using FFTW, Convex, SCS, Plots

# 加载C++库
lib_path = "./libmymath.so"
println("加载C++优化库: $lib_path")

# 模拟你的CPS设置
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

println("原始问题规模: N=$N, M=$M")

# 使用C++库加速矩阵运算
println("\n=== 使用C++库加速计算 ===")

# 1. 使用C++库计算矩阵向量乘法
function matrix_vector_multiply_cpp(matrix, vector)
    rows, cols = size(matrix)
    matrix_flat = vec(matrix)
    result = zeros(rows)

    ccall((:matrix_vector_multiply, lib_path), Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
          matrix_flat, vector, result, rows, cols)
    return result
end

# 2. 使用C++库计算向量点积
function dot_product_cpp(a, b)
    n = length(a)
    return ccall((:dot_product, lib_path), Cdouble,
                 (Ptr{Cdouble}, Ptr{Cdouble}, Cint),
                 a, b, n)
end

# 3. 使用C++库计算L2范数
function l2_norm_cpp(vector)
    n = length(vector)
    return ccall((:l2_norm, lib_path), Cdouble,
                 (Ptr{Cdouble}, Cint),
                 vector, n)
end

# 测试C++库的性能
println("测试C++库性能...")

# 测试矩阵向量乘法
println("矩阵向量乘法测试:")
start_time = time()
for i in 1:100
    result_cpp = matrix_vector_multiply_cpp(K, GFV)
end
cpp_time = time() - start_time

start_time = time()
for i in 1:100
    result_julia = K * GFV
end
julia_time = time() - start_time

println("C++时间: $(cpp_time)秒")
println("Julia时间: $(julia_time)秒")
println("加速比: $(julia_time/cpp_time)")

# 测试向量点积
println("\n向量点积测试:")
a_test = rand(M)
b_test = rand(M)

start_time = time()
for i in 1:1000
    dot_cpp = dot_product_cpp(a_test, b_test)
end
cpp_time = time() - start_time

start_time = time()
for i in 1:1000
    dot_julia = dot(a_test, b_test)
end
julia_time = time() - start_time

println("C++时间: $(cpp_time)秒")
println("Julia时间: $(julia_time)秒")
println("加速比: $(julia_time/cpp_time)")

# 集成到你的CPS求解器中
println("\n=== 集成到CPS求解器 ===")

# 原始的Convex.jl求解
vx = Variable(M)
problem = minimize(norm(vx, 1), K * F * vx == GFV)
result = solve!(problem, SCS.Optimizer)
reAw = evaluate(vx)
reA1 = idct(reAw)

# 使用C++库计算残差
residual_cpp = matrix_vector_multiply_cpp(K, reA1) - GFV
error_cpp = l2_norm_cpp(residual_cpp)

residual_julia = K * reA1 - GFV
error_julia = norm(residual_julia)

println("C++计算的残差: $error_cpp")
println("Julia计算的残差: $error_julia")
println("残差差异: $(abs(error_cpp - error_julia))")

# 可视化结果
println("\n=== 可视化结果 ===")
plot(ctx.mesh, A.(ctx.mesh); xlabel="ω", ylabel="A(ω)", label="original A(ω)",
     title="C++集成测试 - noise = $noise")
plot!(ctx.mesh, reA1; label="Convex.jl + C++库")
plot!(ctx.mesh, real(residual_cpp); label="残差 (C++计算)", alpha=0.5)

println("测试完成！C++库已成功集成到CPS项目中。")
