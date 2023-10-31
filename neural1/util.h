#pragma once 

#include <iostream> 
#include <array>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

using std::cout;
using std::endl;  

template <std::size_t Rows1, std::size_t Cols1, std::size_t Rows2, std::size_t Cols2> //
void matrixMultiply(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols2>, Rows2>& matrix2, std::array<std::array<float, Cols2>, Rows1>& matrix3) { //traditional matrix multiplication
    if (Cols1 != Rows2)
    {
        std::cout << "Cannot multiply matrices of these dimensions" << std::endl;
        return;
    }

    for (std::size_t i = 0; i < Rows1; i++)
    {
        for (std::size_t j = 0; j < Cols2; j++)
        {
            float sum = 0;
            for (std::size_t k = 0; k < Cols1; k++)
            {
                sum += matrix1[i][k] * matrix2[k][j];
            }
            matrix3[i][j] = sum;
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1, std::size_t Rows2, std::size_t Cols2>
void matrixMultiplyTransposeFirstArgument(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols2>, Rows2>& matrix2, std::array<std::array<float, Cols2>, Cols1>& matrix3) { //multiplies the transposition of the first matrix by the the second, doesnt actually alter the first matrix 
    if (Rows1 != Rows2)
    {
        std::cout << "Cannot multiply matrices of these dimensions" << std::endl;
        return;
    }
    for (std::size_t i = 0; i < Cols1; i++)
    {
        for (std::size_t j = 0; j < Cols2; j++)
        {
            float sum = 0;
            for (std::size_t k = 0; k < Rows1; k++)
            {
                sum += matrix1[k][i] * matrix2[k][j];
            }
            matrix3[i][j] = sum;
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1, std::size_t Rows2, std::size_t Cols2>
void matrixMultiplyTransposeSecondElement(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols2>, Rows2>& matrix2, std::array<std::array<float, Rows2>, Rows1>& matrix3) {
    if (Cols1 != Cols2)
    {
        std::cout << "Cannot multiply matrices of these dimensions" << std::endl;
        return;
    }

    for (std::size_t i = 0; i < Rows1; i++)
    {
        for (std::size_t j = 0; j < Rows2; j++)  
        {
            float sum = 0;
            for (std::size_t k = 0; k < Cols1; k++)
            {
                sum += matrix1[i][k] * matrix2[j][k];
            }
            matrix3[i][j] = sum;  
        }
    }
}

//since all three matrices should be of the same size for addition subtraction and the hadmard product, we only need a template for one of them (passing a matrix of incorrect dimensions will result in a runtime error)
template <std::size_t Rows1, std::size_t Cols1>
void matrixAdd(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols1>, Rows1>& matrix2, std::array<std::array<float, Cols1>, Rows1>& resultMatrix) { //adds first two matrices and stores result in resultMatrix
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1>
void matrixAddToArg1(std::array<std::array<float, Cols1>, Rows1>& resultMatrix, const std::array<std::array<float, Cols1>, Rows1>& matrix2) { //adds first two matrices and stores result in first matrix
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] += matrix2[i][j];
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1>
void matrixSubtract(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols1>, Rows1>& matrix2, std::array<std::array<float, Cols1>, Rows1>& resultMatrix) { //subtracts second matrix from first and stores result in resultMatrix
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1>
void matrixSubtractFromArg1(std::array<std::array<float, Cols1>, Rows1>& resultMatrix, const std::array<std::array<float, Cols1>, Rows1>& matrix2) { //subtracts second matrix from first and stores result in first matrix
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] -= matrix2[i][j];
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1>
void hadmardProduct(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols1>, Rows1>& matrix2, std::array<std::array<float, Cols1>, Rows1>& resultMatrix) { //multiplies first two matrices elementwise and stores result in resultMatrix

    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] = matrix1[i][j] * matrix2[i][j];
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1>
void ElementwiseMultiplicationByScalar(std::array<std::array<float, Cols1>, Rows1>& matrix1, const float scalar) {  //multiplies first matrix by scalar and stores result in first matrix

    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            matrix1[i][j] *= scalar;
        }
    }
}

template <std::size_t Rows1, std::size_t Cols1>
void sigmoid(const std::array<std::array<float, Cols1>, Rows1>& matrix,  std::array<std::array<float, Cols1>, Rows1>& result) { //applies sigmoid function to each element of matrix and stores result in resultMatrix
    for (std::size_t i = 0; i < Rows1; ++i) {
        for (std::size_t j = 0; j < Cols1; ++j) {
            result[i][j] = 1.0f / (1.0f + std::exp(-matrix[i][j]));
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1>
void sigmoidDerivative(const std::array<std::array<float, Cols1>, Rows1>& matrix, std::array<std::array<float, Cols1>, Rows1>& result) { //applies derivative of sigmoid function to each element of matrix and stores result in resultMatrix
    float sigmoid_value;
    for (std::size_t i = 0; i < Rows1; ++i) {
        for (std::size_t j = 0; j < Cols1; ++j) {
            sigmoid_value = 1.0f / (1.0f + std::exp(-matrix[i][j]));
            result[i][j] = sigmoid_value * (1.0f - sigmoid_value);
        }
    }
}

template<std::size_t Rows, std::size_t Cols>
void setRandom(std::array<std::array<float, Cols>, Rows>& arr) { //sets each element of the array to a random number between 0 and 0.1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.01, 0.01);

    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            arr[i][j] = static_cast<float>(dis(gen));
        }
    }

}
template<std::size_t Rows, std::size_t Cols>
void setZero(std::array<std::array<float, Cols>, Rows>& arr) { //sets each element of the array to 0

    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            arr[i][j] = 0;
        }
    }
}
template<std::size_t Rows, std::size_t Cols>
void printArray(std::array<std::array<float, Cols>, Rows>& arr) { //prints the array to the console
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

}

template<std::size_t Rows, std::size_t Cols>
void copyVectorToArray(std::array<std::array<float, Cols>, Rows>& arr, const std::vector<std::vector<float>> arr2) { //copies a vector to an array
    if (arr2.size() != Rows || (arr2.size() > 0 && arr2[0].size() != Cols)) {
        std::cout << "Dimension mismatch" << std::endl;
        return;
    }
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            arr[i][j] = arr2[i][j];
        }
    }
}
int32_t reverseInt(int32_t i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int32_t)ch1 << 24) + ((int32_t)ch2 << 16) + ((int32_t)ch3 << 8) + ch4;
}
std::vector<std::vector<std::vector<float>>> readTrainingImages(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open())
    {
        int32_t magicNumber = 0,  cols = 0, rows = 0;
        std::size_t numOfImages = 0; 

        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        file.read((char*)&numOfImages, sizeof(numOfImages));
        numOfImages = reverseInt(numOfImages);

        file.read((char*)&rows, sizeof(rows));
        rows = reverseInt(rows);

        file.read((char*)&cols, sizeof(cols));
        cols = reverseInt(cols);

        std::vector<std::vector<std::vector<float>>> images(numOfImages, std::vector<std::vector<float>>(1, std::vector<float>(rows * cols)));

        for (std::size_t i = 0; i < numOfImages; ++i)
        {
            for (std::size_t j = 0; j < rows * cols; ++j)
            {
                unsigned char pixel;
                file.read((char*)&pixel, sizeof(unsigned char));
                images[i][0][j] = static_cast<float>(pixel) / 255.0f;
            }
        }

        std::cout << "Successfully loaded " << numOfImages << " Images From " << filename << std::endl;
        file.close();
        return images;
    }
    else
    {
        std::cout << "Cannot open file: " << filename << std::endl;
        return std::vector<std::vector<std::vector<float>>>();
    }
}

std::vector<int> readTrainingLabels(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open())
    {
        int32_t magicNumber = 0;
        std::size_t numOfLabels = 0; //number of labels should be the same as the number of images

        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        file.read((char*)&numOfLabels, sizeof(numOfLabels));
        numOfLabels = reverseInt(numOfLabels);

        std::vector<int> labels(numOfLabels);

        for (std::size_t i = 0; i < numOfLabels; ++i)
        {
            unsigned char label;
            file.read((char*)&label, sizeof(unsigned char));
            labels[i] = static_cast<int>(label);
        }

        file.close();
        std::cout << "Successfully loaded " << numOfLabels << " Labels From " << filename << std::endl;
        return labels;
    }
    else
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }
}
