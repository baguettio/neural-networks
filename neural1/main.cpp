#include <iostream> 
#include <array>
//REMEMBER MATRIX ADD IS NOT REPORTING DIMENSION ERRORS PROPERLY
template <std::size_t Rows1, std::size_t Cols1, std::size_t Rows2, std::size_t Cols2, std::size_t Rows3, std::size_t Cols3> //template so that out matrix functions can take std::arrays of all sizes
void matrixMultiply(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols2>, Rows2>& matrix2,  std::array<std::array<float, Cols3>, Rows3>& matrix3) {
    if (Cols1 != Rows2)
    {
        std::cout << "Cannot multiply matrices of these dimensions" << std::endl;
        return;
    }
    else if (Rows3 != Rows1 || Cols3 != Cols2)
    {
        std::cout << "Result matrix is of incorrect dimensions" << std::endl;
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
template <std::size_t Rows1, std::size_t Cols1, std::size_t Rows2, std::size_t Cols2, std::size_t Rows3, std::size_t Cols3>
void matrixAdd(const std::array<std::array<float, Cols1>, Rows1>& matrix1, const std::array<std::array<float, Cols2>, Rows2>& matrix2, std::array<std::array<float, Cols3>, Rows3>& resultMatrix) {
    if(Cols1 != Cols2 || Cols1 != Cols3 || Rows1 != Rows2 || Rows1 != Rows3){
        std::cout << "Invalid dimensions for matrix addition" << std::endl;
    }
    
    for (std::size_t i = 0; i < Rows1; i++) {
        for (std::size_t j = 0; j < Cols1; j++) {
            resultMatrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}
int main(){
    constexpr size_t inputNeurons = 784, hiddenNeurons = 30, outputNeurons = 10;

    std::array<float, inputNeurons> inputLayer; // the input vector
    std::array<float, hiddenNeurons> hiddenBiases; // the biases for the hidden layer
    std::array<std::array<float, inputNeurons>, hiddenNeurons> inputWeights; // the weights connecting the input and hidden layer

    std::array<float, hiddenNeurons> hiddenLayer; // the hidden layer
    std::array<float, outputNeurons> outputBiases; // the biases for the output layer
    std::array<std::array<float, hiddenNeurons>, outputNeurons> hiddenWeights; // the weights connecting the hidden and output layer

    std::array<float, outputNeurons> outputLayer; // the output layer

    std::array<float, outputNeurons> outputError; //error of output layer (used for adjusting biases also)
    std::array<float, hiddenNeurons> hiddenError; //error of hidden layer
    std::array<std::array<float, hiddenNeurons>, outputNeurons> hiddenWeightsPD; //partial derivatvies for the weights
    std::array<std::array<float, inputNeurons>, hiddenNeurons> inputWeightsPD;

    //hyperparameters
    size_t epochs = 10; //number of times we iterate through all of the training data
    size_t batchSize = 128; //how many training examples we look at before we adjust the parameters
    float learningRate = 0.01; //how much we adjust the parameters 


    constexpr size_t Rows1 = 2, Cols1 = 3, Rows2 = 3, Cols2 = 2;

    std::array<std::array<float, Cols1>, Rows1> matrix1 = {
        std::array<float, Cols1>{1, 2, 3},
        std::array<float, Cols1>{4, 5, 6}
    };

    std::array<std::array<float, Cols1>, Rows1> matrix2 = {
        std::array<float, Cols1>{7, 8},
        std::array<float, Cols1>{9, 10},
    };

    std::array<std::array<float, Cols1>, Rows1> result = {};

    matrixAdd(matrix1, matrix2, result);

    for(const auto& row : result) {
        for(const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    return 0;



    return 0;




}
