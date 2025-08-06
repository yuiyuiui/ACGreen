#include <iostream>
#include <vector>
#include <cmath>

extern "C" {
    // 简单的加法函数
    double add(double a, double b) {
        return a + b;
    }
    
    // 乘法函数
    double multiply(double a, double b) {
        return a * b;
    }
    
    // 向量点积函数
    double dot_product(double* a, double* b, int n) {
        double result = 0.0;
        for (int i = 0; i < n; i++) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    // 矩阵向量乘法
    void matrix_vector_multiply(double* matrix, double* vector, double* result, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            result[i] = 0.0;
            for (int j = 0; j < cols; j++) {
                result[i] += matrix[i * cols + j] * vector[j];
            }
        }
    }
    
    // 计算向量的L2范数
    double l2_norm(double* vector, int n) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += vector[i] * vector[i];
        }
        return sqrt(sum);
    }
    
    // 向量加法
    void vector_add(double* a, double* b, double* result, int n) {
        for (int i = 0; i < n; i++) {
            result[i] = a[i] + b[i];
        }
    }
    
    // 向量标量乘法
    void vector_scale(double* vector, double scalar, double* result, int n) {
        for (int i = 0; i < n; i++) {
            result[i] = vector[i] * scalar;
        }
    }
} 