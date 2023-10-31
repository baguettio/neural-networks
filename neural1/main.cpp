#include "util.h"

constexpr size_t inputNeurons = 784, hiddenNeurons = 10, outputNeurons = 10;

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

std::vector<std::vector<std::vector<float>>> testImages;
std::vector<int> testLabels;

std::vector<std::vector<std::vector<float>>> images;
std::vector<int> labels;

int previousAccuracy = 0, currentAccuracy = 0;

constexpr int numOfTrainingImages = 60000, numOfTestingImages = 10000;

bool testDataLoaded = false;
bool trainDataLoaded = false;
void load(){
    if (!trainDataLoaded)
    {
        images = readTrainingImages("train-images.idx3-ubyte"); //read in the training images
        labels = readTrainingLabels("train-labels.idx1-ubyte"); //read in the training labels
        trainDataLoaded = true;
    }
    else
    {
        cout << "Training data already loaded" << endl;
    }
    if (!testDataLoaded)
    {
        testLabels = readTrainingLabels("t10k-labels.idx1-ubyte"); // read in the testing labels
        testImages = readTrainingImages("t10k-images.idx3-ubyte"); //read in the testing images
        testDataLoaded = true;
    }
    else
    {
        cout << "Testing data already loaded" << endl;
	}
}
void feedForward(std::vector<std::vector<float>>& currentImage) {
    copyVectorToArray(inputLayer, currentImage);
    matrixMultiply(inputLayer, inputWeights, z1);
    matrixAdd(z1, hiddenBiases, z1);
    sigmoid(z1, hiddenLayer);
    matrixMultiply(hiddenLayer, hiddenWeights, z2);
    matrixAdd(z2, outputBiases, z2);
    sigmoid(z2, outputLayer);
}
void testNetwork() {
    if (!testDataLoaded) 
    {
        cout << "Test data not loaded" << endl;
		return;
    }
    int correctC = 0;
    for (std::size_t current_image = 0; current_image < 10000; ++current_image) //test the accuracy of the network on the test data
    {
        int label = testLabels[current_image];
        feedForward(testImages[current_image]);
        auto it = std::max_element(outputLayer[0].begin(), outputLayer[0].end());
        int index = std::distance(outputLayer[0].begin(), it);
        if (index == label) correctC++;
        
    }
    previousAccuracy = correctC;
    currentAccuracy = correctC;
    cout << "Correct : " << correctC << "/" << 10000 << " " << static_cast<float>(correctC) / static_cast<float>(10000) << endl;
}
int main() {

    size_t epochs = 100; //number of times we iterate through all of the training data
    float learningRate = 0.05; //how much we adjust the parameters by

    setRandom(inputWeights);
    setRandom(hiddenWeights);
    setZero(hiddenBiases);
    setZero(outputBiases);

    load();
    testNetwork();

    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
        for (std::size_t current_image = 0; current_image < numOfTrainingImages; ++current_image)
        {

            setZero(outputError);
            setZero(hiddenWeightsPD);
            setZero(hiddenError);
            setZero(inputWeightsPD);
            copyVectorToArray(inputLayer, images[current_image]);

            int label = labels[current_image];
            std::array<std::array<float, 10>, 1> correct = { 0 };
            correct[0][label] = 1;

            //forwards pass
            feedForward(images[current_image]);

            //backwards pass
            matrixSubtract(outputLayer, correct, outputError); //get the difference between the correct output and the actual output
            ElementwiseMultiplicationByScalar(outputError, 2); //multiply by two to get the derivative of MSE with respect to the cost
            sigmoidDerivative(z2,z2); //get the derivative of the activation function with respect to the cost
            hadmardProduct(outputError, z2, outputError); //multiply the two together to get the derivative of the output function with respect to the cost (outputError)

            matrixMultiplyTransposeFirstArgument(hiddenLayer, outputError, hiddenWeightsPD); //multiply the hidden layer by the output error to get the partial derivatives for the hidden weights)

            matrixMultiplyTransposeSecondElement(outputError, hiddenWeights, hiddenError); //multiply the output error by the transposed hidden weights to get the backpropagated error (storing it in the hidden error array to save memory)
            sigmoidDerivative(z1,z1); //get the derivative of the sigmoid function with respect to the weighted sum (z1)
            hadmardProduct(hiddenError, z1, hiddenError); //multiply the backpropagated error by the derivative of the activation function to get the derivative of the hidden layer with respect to the cost (hidden Error)

            matrixMultiplyTransposeFirstArgument(inputLayer, hiddenError, inputWeightsPD); //multiply the input layer by the hidden error to get the partial derivatives with respect to the cost for the input weights

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
        //learning rate schedule
        if (epoch % 10 == 0)
		{
			learningRate *= 0.9;
		}


  

  

    }
    testNetwork();
    return 0;
}



