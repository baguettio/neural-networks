#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <fstream>
#include <array>
#include <cstdint>

template <std::size_t Rows1, std::size_t Cols1, std::size_t Rows2, std::size_t Cols2> 
void matrixMultiply(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols2>, Rows2>& matrix2,  std::array<std::array<float, Cols2>, Rows1>& matrix3) {
    if (Cols1 != Rows2)
    {
        std::cout << "Cannot multiply matrices of these dimensions" << std::endl;
        return;
    }

    for(std::size_t i = 0; i < Rows1; i++)
    {
        for(std::size_t j = 0; j < Cols2; j++)
        {
            float sum = 0;
            for(std::size_t k = 0; k < Cols1; k++)
            {
                sum += matrix1[i][k] * matrix2[k][j];
            }
            matrix3[i][j] = sum;
        }
    }
}
//since all three matrices should be of the same size for addition subtraction and the hadmard product, we only need a template for one of them (passing a matrix of incorrect dimensions will result in a runtime error)
template <std::size_t Rows1, std::size_t Cols1>
void matrixAdd(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols1>, Rows1>& matrix2, std::array<std::array<float, Cols1>, Rows1>& resultMatrix) {
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1>
void matrixSubtract(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols1>, Rows1>& matrix2, std::array<std::array<float, Cols1>, Rows1>& resultMatrix) {
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1>
void hadmardProduct(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols1>, Rows1>& matrix2, std::array<std::array<float, Cols1>, Rows1>& resultMatrix) {
    
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] = matrix1[i][j] * matrix2[i][j];
        }
    }
}
//and for transposition, we still only need one template but the second matrix will be of size cols1 x rows1
template <std::size_t Rows1, std::size_t Cols1>
void transpose(const std::array<std::array<float, Cols1>, Rows1>& matrix1, std::array<std::array<float, Rows1>, Cols1>& resultMatrix) {
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[j][i] = matrix1[i][j];
        }
    }
}

constexpr int numOfTrainingImages = 60000;
constexpr int numOfTestingImages = 10000;

std::array<std::array<std::array<float, 784>, 1>, numOfTrainingImages> readTrainingImages (const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary); //open a file in binary mode
    std::array<std::array<std::array<float, 784>, 1>, numOfTrainingImages> datset;
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(1);
    }
    file.seekg(16); //skip the 16 header bits
    for(std::size_t i = 0; i < numOfTrainingImages; i++){

    }

}
#endif // UTIL_H