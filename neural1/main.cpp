#include "util.h"

constexpr size_t inputNeurons = 784, hiddenNeurons = 10, outputNeurons = 10;
constexpr int numOfTrainingImages = 6000, numOfTestingImages = 1000;
bool testDataLoaded = false, trainDataLoaded = false;
//vectors to store images and lables in (using an std::array will cause a stack overflow)
std::vector<std::vector<std::vector<float>>> testImages;
std::vector<int> testLabels;
std::vector<std::vector<std::vector<float>>> images;
std::vector<int> labels;
class Network
{
    public:
        std::array<std::array<float, inputNeurons>, 1> inputLayer; // the input vector
        std::array<std::array<float, hiddenNeurons>, 1> hiddenBiases; // the biases for the hidden layer
        std::array<std::array<float, hiddenNeurons>, inputNeurons> inputWeights; // the weights connecting the input and hidden layer
        std::array<std::array<float, hiddenNeurons>, 1> hiddenLayer; // the hidden layer
        std::array<std::array<float, outputNeurons>, 1> outputBiases; // the biases for the output layer
        std::array<std::array<float, outputNeurons>, hiddenNeurons> hiddenWeights; // the weights connecting the hidden and output layer
        std::array<std::array<float, outputNeurons>, 1> outputLayer; // the output layer
        std::array<std::array<float, outputNeurons>, 1> outputError; //error of output layer (used for adjusting biases and calculating weights PD)
        std::array<std::array<float, hiddenNeurons>, 1> hiddenError; //error of hidden layer
        std::array<std::array<float, outputNeurons>, hiddenNeurons> hiddenWeightsPD; //partial derivatvies for the weights
        std::array<std::array<float, hiddenNeurons>, inputNeurons> inputWeightsPD;
        std::array<std::array<float, hiddenNeurons>, 1> z1; //intermediate layers for derivative calculations
        std::array<std::array<float, outputNeurons>, 1> z2;
        int currentAccuracy = -1, previousAccuracy = 0, difference = 0;
        float learning_rate = 0.01;
        int count = 0;
        int margin = 5;

        void initalize()
        {
            setRandom(inputWeights);
            setRandom(hiddenWeights);
            setZero(hiddenBiases);
            setZero(outputBiases);
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

        void test() 
        {
            
            if (!testDataLoaded) 
            {
                cout << "Test data not loaded" << endl;
                return;
            }

            int correctC = 0;
            for (std::size_t current_image = 0; current_image < numOfTestingImages; ++current_image) //test the accuracy of the network on the test data
            {
                int label = testLabels[current_image];
                feedForward(testImages[current_image]);
                auto it = std::max_element(outputLayer[0].begin(), outputLayer[0].end());
                int index = std::distance(outputLayer[0].begin(), it);
                if (index == label) correctC++;
                
            }
            previousAccuracy = currentAccuracy;
            currentAccuracy = correctC;
            difference = currentAccuracy - previousAccuracy;
            cout << "Correct : " << correctC << "/" << 10000 << " " << static_cast<float>(correctC) / static_cast<float>(10000) << endl;
        }    
        void backpropagateAndOveride(int label)
        {
            std::array<float, 10> correct = {0};
            correct[label] = 1;
            matrixSubtract(outputLayer, correct, outputError); //get the difference between the correct output and the actual output
            ElementwiseMultiplicationByScalar(outputError, 2); //multiply by two to get the derivative of MSE with respect to the cost
            sigmoidDerivative(z2,z2); //get the derivative of the activation function with respect to the cost
            hadmardProduct(outputError, z2, outputError); //multiply the two together to get the derivative of the output function with respect to the cost (outputError)
            matrixMultiplyTransposeFirstArgument(hiddenLayer, outputError, hiddenWeightsPD); //multiply the hidden layer by the output error to get the partial derivatives for the hidden weights)
            matrixMultiplyTransposeSecondElement(outputError, hiddenWeights, hiddenError); //multiply the output error by the transposed hidden weights to get the backpropagated error (storing it in the hidden error array to save memory)
            sigmoidDerivative(z1,z1); //get the derivative of the sigmoid function with respect to the weighted sum (z1)
            hadmardProduct(hiddenError, z1, hiddenError); //multiply the backpropagated error by the derivative of the activation function to get the derivative of the hidden layer with respect to the cost (hidden Error)
            matrixMultiplyTransposeFirstArgument(inputLayer, hiddenError, inputWeightsPD); //multiply the input layer by the hidden error to get the partial derivatives with respect to the cost for the input weights

        }       
        void backpropagateAndSum(int label)
        {
            //TODO
            cout << "Not implemented yet" << endl;
            return;
        }       
        void adjust()
        {
            //adjust partial derivatives/costs by learning rate
            ElementwiseMultiplicationByScalar(outputError, learning_rate);
            ElementwiseMultiplicationByScalar(hiddenWeightsPD, learning_rate);
            ElementwiseMultiplicationByScalar(hiddenError, learning_rate);
            ElementwiseMultiplicationByScalar(inputWeightsPD, learning_rate);

            //adjust parameters by the adjusted partial derivatives
            matrixSubtractFromArg1(outputBiases, outputError);
            matrixSubtractFromArg1(hiddenWeights, hiddenWeightsPD);
            matrixSubtractFromArg1(hiddenBiases, hiddenError);
            matrixSubtractFromArg1(inputWeights, inputWeightsPD);

        } 
        bool exitEarly()
        {
            return previousAccuracy >= currentAccuracy;
        }
        void learning_rate_scheduler()
        {
            if(difference < margin)//if we're barely improving or getting worse, turn the learning rate way down
            {
                cout << "Reached margin" << endl;
                learning_rate = 0.0001; 
            }
        }                                                                                                               
};
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

int main() {
    Network classifier;

    size_t epochs = 100; //number of times we iterate through all of the training data
    classifier.learning_rate = 0.1; //scale factor of how much we adjust the parameters by
    load();

    classifier.initalize();
    classifier.test();

    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
        for (std::size_t current_image = 0; current_image < numOfTrainingImages; ++current_image)
        {
            //feedforward
            classifier.feedForward(images[current_image]);
            //backwards pass
            classifier.backpropagateAndOveride(labels[current_image]);
            //adjust parameters
            classifier.adjust();
            
        }
        classifier.test();

    }
    classifier.test();
    return 0;
}



