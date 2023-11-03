#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <fstream>
#include <array>
#include <cstdint>
#include <vector>
#include <random>

using std::cout;
using std::endl;
void test();

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
void matrixMultiplyTransposeFirstArgument(const std::array<std::array<float, Cols1>, 
Rows1>& matrix1, const std::array<std::array<float, Cols2>, Rows2>& matrix2,  std::array<std::array<float, Cols2>, Cols1>& matrix3) {
    //cols1 x rows1
    //rows2 x cols2
    //so result is cols1 x cols2
    //and rows1 must equal rows2

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
void matrixMultiplyTransposeSecondElement(const std::array<std::array<float, Cols1>, Rows1>& matrix1, 
const std::array<std::array<float, Cols2>, Rows2>& matrix2,  std::array<std::array<float, Rows2>, Rows1>& matrix3) {

    if (Cols1 != Cols2)
    {
        std::cout << "Cannot multiply matrices of these dimensions" << std::endl;
        return;
    }

    for(std::size_t i = 0; i < Rows1; i++)
    {
        for(std::size_t j = 0; j < Rows2; j++)
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

//since all three matrices should be of the same size for addition subtraction and the hadamard product, we only need a template for one of them (passing a matrix of incorrect dimensions will result in a runtime error)
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
void matrixSubtract(const std::array<std::array<float, Cols1>, Rows1>& matrix1,
 std::array<float, Cols1> matrix2, 
 std::array<std::array<float, Cols1>, Rows1>& resultMatrix) { //subtracts second matrix from first and stores result in resultMatrix
    //as this function is only used for subtracting the predicted output from the correct output, the second matrix is a 1D array, and the second matrix is a 2D array with one row (so we treat it like a 1D array)
    for (std::size_t j = 0; j < Cols1; j++) {
        resultMatrix[0][j] = matrix1[0][j] - matrix2[j]; //only one row in each matrix even if the first is a 2D array
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
void hadamardProduct(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols1>, Rows1>& matrix2, std::array<std::array<float, Cols1>, Rows1>& resultMatrix) {
    
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
    std::uniform_real_distribution<> dis(-0.1, 0.1);

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
void copyVectorToArray(std::array<std::array<float, Cols>, Rows>& arr, const std::vector<std::vector<float>> arr2) { //copies a vector to an array
    if (arr2.size() != Rows || (arr2.size() > 0 && arr2[0].size() != Cols)) {
        std::cout << "Dimension mismatch, arr1 is " << Rows << " by " <<  Cols << " and arr2 is " << arr2.size() << " by " << arr2[0].size() << std::endl;
        return;
    }
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            arr[i][j] = arr2[i][j];
        }
    }
}

constexpr int numOfTrainingImages = 60000, numOfTestingImages = 10000, numOfPixels = 784;

// Function to read images from MNIST dataset
std::vector<std::vector<std::vector<float>>> readMNISTImages(const std::string& filename, size_t numOfImages, size_t rows, size_t cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(1);
    }

    // skip magic number and the number of images, as we're passing them as arguments
    file.seekg(16);

    std::vector<std::vector<std::vector<float>>> images(numOfImages, std::vector<std::vector<float>>(1, std::vector<float>(rows * cols)));

    for (size_t i = 0; i < numOfImages; ++i) {
        for (size_t j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][0][j] = static_cast<float>(pixel) / 255.0f;
        }
    }
    file.close();
    return images;
}

// read labels in similar fashion
std::vector<int> readMNISTLabels(const std::string& filename, size_t numOfImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(1);
    }

    // Skip the magic number and the number of items
    file.seekg(8);

    std::vector<int> labels(numOfImages);
    for (size_t i = 0; i < numOfImages; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = label;
    }
    file.close();
    return labels;
}


#endif // UTIL_H