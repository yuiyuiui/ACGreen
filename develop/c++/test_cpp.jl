# Julia测试文件：演示如何调用C++库
using LinearAlgebra

# 编译C++库（如果还没有编译）
println("正在编译C++库...")
run(`make`)

# 加载动态库
lib_path = "./libmymath.so"
println("加载C++库: $lib_path")

# 测试基本函数
println("\n=== 测试基本函数 ===")
result1 = ccall((:add, lib_path), Cdouble, (Cdouble, Cdouble), 3.14, 2.86)
println("3.14 + 2.86 = $result1")

result2 = ccall((:multiply, lib_path), Cdouble, (Cdouble, Cdouble), 3.14, 2.86)
println("3.14 * 2.86 = $result2")

# 测试向量操作
println("\n=== 测试向量操作 ===")
n = 5
a = [1.0, 2.0, 3.0, 4.0, 5.0]
b = [2.0, 3.0, 4.0, 5.0, 6.0]

# 点积
dot_result = ccall((:dot_product, lib_path), Cdouble,
                   (Ptr{Cdouble}, Ptr{Cdouble}, Cint),
                   a, b, n)
println("向量点积: $dot_result")
println("Julia验证: $(dot(a, b))")

# L2范数
norm_result = ccall((:l2_norm, lib_path), Cdouble,
                    (Ptr{Cdouble}, Cint),
                    a, n)
println("L2范数: $norm_result")
println("Julia验证: $(norm(a))")

# 向量加法
result = zeros(n)
ccall((:vector_add, lib_path), Cvoid,
      (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint),
      a, b, result, n)
println("向量加法: $result")
println("Julia验证: $(a + b)")

# 向量标量乘法
scaled = zeros(n)
ccall((:vector_scale, lib_path), Cvoid,
      (Ptr{Cdouble}, Cdouble, Ptr{Cdouble}, Cint),
      a, 2.5, scaled, n)
println("向量标量乘法: $scaled")
println("Julia验证: $(2.5 * a)")

# 测试矩阵向量乘法
println("\n=== 测试矩阵向量乘法 ===")
rows, cols = 3, 4
matrix = [1.0 2.0 3.0 4.0;
          5.0 6.0 7.0 8.0;
          9.0 10.0 11.0 12.0]
vector = [1.0, 2.0, 3.0, 4.0]

# 将矩阵展平为一维数组
matrix_flat = vec(matrix)

# C++计算
cpp_result = zeros(rows)
ccall((:matrix_vector_multiply, lib_path), Cvoid,
      (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
      matrix_flat, vector, cpp_result, rows, cols)
println("C++矩阵向量乘法: $cpp_result")

# Julia验证
julia_result = matrix * vector
println("Julia验证: $julia_result")

# 性能比较
println("\n=== 性能比较 ===")
using BenchmarkTools

# 准备大数据
n_large = 1000
a_large = rand(n_large)
b_large = rand(n_large)
result_large = zeros(n_large)

# 测试C++向量加法性能
println("C++向量加法性能:")
@btime ccall((:vector_add, lib_path), Cvoid,
             (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint),
             a_large, b_large, result_large, n_large)

# 测试Julia向量加法性能
println("Julia向量加法性能:")
@btime a_large + b_large

println("\n=== 测试完成 ===")
