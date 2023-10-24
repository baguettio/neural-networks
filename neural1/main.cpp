#include "util.h"
#include <iostream> 
#include <array>
#include <fstream>
#include <cstdint>
#include <cmath>

int main(){
    constexpr size_t inputNeurons = 784, hiddenNeurons = 30, outputNeurons = 10;

    static std::array<std::array<float, inputNeurons>, 1> inputLayer; // the input vector
    static std::array<std::array<float, hiddenNeurons>, 1> hiddenBiases; // the biases for the hidden layer
    static std::array<std::array<float, hiddenNeurons>, inputNeurons> inputWeights; // the weights connecting the input and hidden layer

    static std::array<std::array<float, hiddenNeurons>, 1> hiddenLayer; // the hidden layer
    static std::array<std::array<float, outputNeurons>, 1> outputBiases; // the biases for the output layer
    static std::array<std::array<float, outputNeurons>, hiddenNeurons> hiddenWeights; // the weights connecting the hidden and output layer

    static std::array<std::array<float, outputNeurons>, 1> outputLayer; // the output layer

    static std::array<std::array<float, outputNeurons>, 1> outputError; //error of output layer (used for adjusting biases and calculating weights PD)
    static std::array<std::array<float, hiddenNeurons>, 1> hiddenError; //error of hidden layer
    static std::array<std::array<float, outputNeurons>, hiddenNeurons> hiddenWeightsPD; //partial derivatvies for the weights
    static std::array<std::array<float, hiddenNeurons>, inputNeurons> inputWeightsPD;

    static std::array<std::array<float, hiddenNeurons>, 1> z1; //intermediate layers for derivative calculations
    static std::array<std::array<float, outputNeurons>, 1> z2;

    static std::array<std::array<float, hiddenNeurons>, 1> z3; // more temps cus im dumb

    setRandom(inputWeights);
    setRandom(hiddenWeights);


    //hyperparameters
    size_t numOfImages = 100;
    size_t epochs = 3; //number of times we iterate through all of the training data
    size_t batchSize = 50   ; //how many training examples we look at before we adjust the parameters
    size_t batches = numOfImages / batchSize;
    float learningRate = 0.01; //how much we adjust the parameters by
    
    constexpr int numOfTrainingImages = 60000;
    constexpr int numOfTestingImages = 10000;

    using Image = std::array<std::array<float, inputNeurons>, 1>;
    
    static std::array<Image, numOfTrainingImages> images;
    images = readTrainingImages("train-images.idx3-ubyte");
    static std::array<int, numOfTrainingImages> labels;
    labels = readTrainingLabels("train-labels.dix1-ubyte");

    double average_cost = 0;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
        for (std::size_t batch = 0; batch < batches; ++batch)
        {
            std::size_t startImage = batch * batchSize;
            std::size_t endImage = startImage + batchSize;

            for (std::size_t current_image = startImage; current_image < endImage; ++current_image)
            {
                
                inputLayer = images[current_image];
                int label = labels[current_image];
                std::array<std::array<float, 10>, 1> correct = {0};
                correct[0][label] = 1;

                //forwards pass
                matrixMultiply(inputLayer, inputWeights, z1);
                matrixAddToArg1(z1, hiddenBiases);
                ReLU(z1, hiddenLayer);
                matrixMultiply(hiddenLayer, hiddenWeights, z2);
                matrixAddToArg1(z2, outputBiases);
                ReLU(z2, outputLayer);
                
                
                //report cost
                for(std::size_t i = 0; i < outputNeurons; ++i){
                    average_cost += pow(correct[0][i] - outputLayer[0][i], 2);
                }

                //backwards pass
                //calculate output erorr
                matrixSubtract(correct, outputLayer, outputError); 
                ReLUDerivative(z2, z2);
                hadmardProduct(outputError, z2, outputError);
                //use output error to calculate the partial derivatives for the hidden weights
                matrixMultiplyTransposeFirstArgument(hiddenLayer, outputError, hiddenWeightsPD); 
                //use output erorr to calculate hidden error
                ReLUDerivative(z1,z1);
                matrixMultiplyTransposeSecondElement(outputError, hiddenWeights, z3);
                hadmardProduct(z3, z1, hiddenError);
                //use hidden error to calculate partial derivatives for the input weights
                matrixMultiplyTransposeFirstArgument(inputLayer, hiddenError, inputWeightsPD);

                matrixSubtractFromArg1(outputBiases, outputError);
                matrixSubtractFromArg1(hiddenWeights, hiddenWeightsPD);
                matrixSubtractFromArg1(outputBiases, outputError);
                matrixSubtractFromArg1(inputWeights, inputWeightsPD);

            }
        }

    }
    std::cout << "Average Cost this epoch is " << average_cost / numOfTrainingImages << std::endl;
    average_cost = 0;

    return 0;

}
