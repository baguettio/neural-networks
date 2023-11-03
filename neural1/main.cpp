#include "util.h"
#include <iostream> 
#include <array>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <algorithm>

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
void feedForward(std::vector<std::vector<float>>& currentImage)
{
    copyVectorToArray(inputLayer, currentImage);
    matrixMultiply(inputLayer, inputWeights, z1);
    matrixAddToArg1(z1, hiddenBiases);
    ReLU(z1, hiddenLayer);
    matrixMultiply(hiddenLayer, hiddenWeights, z2);
    matrixAddToArg1(z2, outputBiases);
    ReLU(z2, outputLayer);
}

std::vector<std::vector<std::vector<float>>> testImages;
std::vector<int> testLabels;
int main(){


    setRandom(inputWeights);
    setRandom(hiddenWeights);
    setZero(hiddenBiases);
    setZero(outputBiases);


    //hyperparameters

    size_t epochs = 50; //number of times we iterate through all of the training data
    float learningRate = 0.01; //how much we adjust the parameters by

    


    //load and store traning and testing images/labels
    cout << "Loading images and labels...";
    std::vector<std::vector<std::vector<float>>> images = readMNISTImages("train-images.idx3-ubyte", numOfTrainingImages, 28, 28);
    std::vector<int> labels = readMNISTLabels("train-labels.idx1-ubyte", numOfTrainingImages);
    testImages = readMNISTImages("t10k-images.idx3-ubyte", numOfTestingImages, 28, 28);
    testLabels = readMNISTLabels("t10k-labels.idx1-ubyte", numOfTestingImages);
    cout << "Done" << endl;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
    
        for (std::size_t current_image = 0; current_image < numOfTrainingImages; ++current_image)
        {

            setZero(outputError);
            setZero(hiddenWeightsPD);
            setZero(hiddenError);
            setZero(inputWeightsPD);

            int label = labels[current_image];
            std::array<float, 10> correct = {0};
            correct[label] = 1;

            //forwards pass
            feedForward(images[current_image]);
            
            //backwards pass
            //calculate output erorr
            matrixSubtract(outputLayer, correct, outputError); 
            ReLUDerivative(z2, z2);
            hadamardProduct(outputError, z2, outputError);

            //use output error to calculate the partial derivatives for the hidden weights
            matrixMultiplyTransposeFirstArgument(hiddenLayer, outputError, hiddenWeightsPD); 

            //use output erorr to calculate hidden error
            ReLUDerivative(z1,z1);
            matrixMultiplyTransposeSecondElement(outputError, hiddenWeights, z3);
            hadamardProduct(z3, z1, hiddenError);

            //use hidden error to calculate partial derivatives for the input weights
            matrixMultiplyTransposeFirstArgument(inputLayer, hiddenError, inputWeightsPD);

            //adjust partial derivatives/costs by learning rate

            ElementwiseMultiplicationByScalar(outputError, learningRate);
            ElementwiseMultiplicationByScalar(hiddenWeightsPD, learningRate);
            ElementwiseMultiplicationByScalar(hiddenError, learningRate);
            ElementwiseMultiplicationByScalar(inputWeightsPD, learningRate);

            //adjust parameters by the adjusted partial derivatives
            matrixSubtractFromArg1(outputBiases, outputError);
            matrixSubtractFromArg1(hiddenWeights, hiddenWeightsPD);
            matrixSubtractFromArg1(hiddenBiases, hiddenError);
            matrixSubtractFromArg1(inputWeights, inputWeightsPD);


        }
        test();

    }


    return 0;

}

//function to test the network on the test data
void test() 
{

    int correct_count = 0;
    for (std::size_t current_image = 0; current_image < numOfTestingImages; ++current_image)
    {
        feedForward(testImages[current_image]);
        int label = testLabels[current_image];
        std::array<float, 10> correct = {0};
        correct[label] = 1;
        if (outputLayer[0][label] == *std::max_element(outputLayer[0].begin(), outputLayer[0].end())) {
            correct_count++;
        }
    }
    cout << "Accuracy: " << correct_count << "/" << numOfTestingImages << endl;
}
