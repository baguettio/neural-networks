#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <fstream>
#include <array>
#include <cstdint>
#include <vector>

#include <random>

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
template <std::size_t Rows1, std::size_t Cols1, std::size_t Rows2, std::size_t Cols2> 
void matrixMultiplyTransposeFirstArgument(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols2>, Rows2>& matrix2,  std::array<std::array<float, Cols2>, Cols1>& matrix3) {
    
    if (Rows1 != Rows2)
    {
        std::cout << "Cannot multiply matrices of these dimensions" << std::endl;
        return;
    }

    for(std::size_t i = 0; i < Cols1; i++)
    {
        for(std::size_t j = 0; j < Cols2; j++)
        {
            float sum = 0;
            for(std::size_t k = 0; k < Rows1; k++)
            {
                sum += matrix1[k][i] * matrix2[k][j];
            }
            matrix3[i][j] = sum;
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1, std::size_t Rows2, std::size_t Cols2> 
void matrixMultiplyTransposeSecondElement(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols2>, Rows2>& matrix2,  std::array<std::array<float, Rows2>, Rows1>& matrix3) {
    if (Cols1 != Cols2)
    {
        std::cout << "Cannot multiply matrices of these dimensions" << std::endl;
        return;
    }

    for(std::size_t i = 0; i < Rows1; i++)
    {
        for(std::size_t j = 0; j < Rows1; j++)
        {
            float sum = 0;
            for(std::size_t k = 0; k < Cols1; k++)
            {
                sum += matrix1[i][k] * matrix2[j][k];
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
void matrixAddToArg1(std::array<std::array<float, Cols1>, Rows1>& resultMatrix, const std::array<std::array<float, Cols1>, Rows1>& matrix2) {
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] += matrix2[i][j];
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
void matrixSubtractFromArg1(std::array<std::array<float, Cols1>, Rows1>& resultMatrix, const std::array<std::array<float, Cols1>, Rows1>& matrix2) {
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] -= matrix2[i][j];
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
template <std::size_t Rows1, std::size_t Cols1>
void ElementwiseMultiplicationByScalar(std::array<std::array<float, Cols1>, Rows1>& matrix1, const float scalar) {
    
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            matrix1[i][j] = (matrix1[i][j] * scalar);
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1>
void ReLU(std::array<std::array<float, Cols1>, Rows1>& matrix, std::array<std::array<float, Cols1>, Rows1>& result) {
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            result[i][j] = matrix[i][j] * (matrix[i][j] > 0);
        }
    }
}
template <std::size_t Rows1, std::size_t Cols1>
void ReLUDerivative(std::array<std::array<float, Cols1>, Rows1>& matrix, std::array<std::array<float, Cols1>, Rows1>& result) {
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            result[i][j] = (matrix[i][j] > 0);
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
template<std::size_t Rows, std::size_t Cols>
void setRandom(std::array<std::array<float, Cols>, Rows>& arr) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            arr[i][j] = static_cast<float>(dis(gen));
        }
    }
}
template<std::size_t Rows, std::size_t Cols>
void setZero(std::array<std::array<float, Cols>, Rows>& arr) {

    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            arr[i][j] = 0;
        }
    }
}
template<std::size_t Rows, std::size_t Cols>
void printArray(std::array<std::array<float, Cols>, Rows>& arr) {
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
}

template<std::size_t Rows, std::size_t Cols>
void copyVectorToArray(std::array<std::array<float, Cols>, Rows>& arr, const std::vector<std::vector<float>> arr2) {

    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            arr[i][j] = arr2[i][j];
        }
    }
}

constexpr int numOfTrainingImages = 60000, numOfTestingImages = 10000, numOfPixels = 784;

std::vector<std::vector<std::vector<float>>> readTrainingImages(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary); //open a file in binary mode
    std::vector<std::vector<std::vector<float>>> dataset(numOfTrainingImages, std::vector<std::vector<float>>(1, std::vector<float>(numOfPixels, 0)));

    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(1);
    }
    file.seekg(16); //skip the 16 header bits
    for(std::size_t i = 0; i < numOfTrainingImages; ++i){
        uint8_t buffer[784];
        file.read(reinterpret_cast<char*>(buffer), 784); //read a "squashed" image to a buffer array, treating the buffer array as if it was made up of char*s
        for (int j = 0; j < numOfPixels; ++j) {
            dataset[i][0][j] = static_cast<float>(buffer[j]) / 255.0; //and then iterate through this array, cast it to a float divide by 255 (to normalise between 0 and 1) and set that element of the dataset to it
        }
    }
    file.close();
    return dataset;
}
std::array<int, numOfTrainingImages> readTrainingLabels (const std::string& filename)
{
   std::ifstream file(filename, std::ios::binary); //open the librairy file in binary read mode
    std::array<int, numOfTrainingImages> labels; //create a array to store the labels in
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(1);
    }
    file.seekg(8); //skip header
    uint8_t label = 0;
    for (std::size_t i = 0; i < numOfTrainingImages; ++i) {
        file.read(reinterpret_cast<char*>(&label), 1); //read the current label (as a uint8_t) to our label variable
        labels[i] = static_cast<int>(label); //copy across as a int to array of labels
    }
    file.close();
    return labels;
}


#endif // UTIL_H