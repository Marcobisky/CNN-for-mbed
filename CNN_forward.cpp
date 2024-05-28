#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 15
#define CONV_OUT_CHANNELS 2
#define CONV_KERNEL_SIZE 3
#define CONV_STRIDE 1
#define CONV_PADDING 1
#define POOL_SIZE 2
#define FC1_SIZE 4
#define OUTPUT_SIZE 2

// Define the network parameters based on the provided weights and biases
float conv1_weight[CONV_OUT_CHANNELS][1][CONV_KERNEL_SIZE] = {
    {{3.352316617965698242e-01, 1.919706344604492188e+00, -6.904200911521911621e-01}},
    {{-3.325339639559388161e-04, -1.859567523002624512e+00, 2.665684223175048828e+00}}
};

float conv1_bias[CONV_OUT_CHANNELS] = {-1.660052299499511719e+00, -3.263219356536865234e+00};

float fc1_weight[FC1_SIZE][CONV_OUT_CHANNELS * 7] = {
    {-1.212380051612854004e+00, 4.897625446319580078e-01, 1.676752418279647827e-01, -1.524278372526168823e-01, -4.255052506923675537e-01, -2.748670578002929688e-01, 2.595482394099235535e-03, 8.271283507347106934e-01, -2.903302907943725586e-01, -1.322476506233215332e+00, -2.882294654846191406e-01, -4.944142699241638184e-02, -6.460737586021423340e-01, 1.248078465461730957e+00},
    {-8.875539898872375488e-01, 5.746094137430191040e-02, -3.839503228664398193e-01, -1.314547896385192871e+00, -1.471168756484985352e+00, -1.596968919038772583e-01, 2.556749805808067322e-02, 1.866063594818115234e+00, -8.968020975589752197e-02, -1.357353687286376953e+00, -1.324755907058715820e+00, -5.692279934883117676e-01, 9.027542471885681152e-01, 2.145124197006225586e+00},
    {5.383568406105041504e-01, -2.173725217580795288e-01, 4.004563204944133759e-03, -9.980111122131347656e-01, 6.637406349182128906e-01, 4.259735047817230225e-01, 6.733140945434570312e-01, -2.853965163230895996e-01, 1.044558405876159668e+00, 2.880272269248962402e-01, -7.011796832084655762e-01, -8.295958489179611206e-02, 1.356191754341125488e+00, 1.012199074029922485e-01},
    {2.122551947832107544e-01, 1.777093112468719482e-01, 4.079014956951141357e-01, 3.937144875526428223e-01, 9.601281583309173584e-02, 7.248925417661666870e-02, 4.856705963611602783e-01, 2.153680175542831421e-01, -1.950401253998279572e-02, -1.429779827594757080e-01, -2.198010534048080444e-01, 5.066350102424621582e-02, 1.233443737030029297e+00, 1.934653669595718384e-01}
};

float fc1_bias[FC1_SIZE] = {3.232861280441284180e+00, -3.131519258022308350e-01, -5.868179202079772949e-01, -2.497493267059326172e+00};

float fc2_weight[OUTPUT_SIZE][FC1_SIZE] = {
    {2.663643598556518555e+00, -2.575386285781860352e+00, -1.199447274208068848e+00, 2.540076494216918945e+00},
    {-2.443423032760620117e+00, 2.260564327239990234e+00, 1.291208744049072266e+00, -2.878274440765380859e+00}
};

float fc2_bias[OUTPUT_SIZE] = {-3.020233154296875000e+00, 2.527628421783447266e+00};

// Activation functions
float relu(float x) {
    return x > 0 ? x : 0;
}

void softmax(float* input, int length, float* output) {
    float max = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// Forward propagation function
void forward(float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {
    // Convolutional layer
    float conv_out[CONV_OUT_CHANNELS][INPUT_SIZE];
    for (int i = 0; i < CONV_OUT_CHANNELS; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            conv_out[i][j] = conv1_bias[i];
            for (int k = 0; k < CONV_KERNEL_SIZE; k++) {
                int idx = j - CONV_PADDING + k;
                if (idx >= 0 && idx < INPUT_SIZE) {
                    conv_out[i][j] += conv1_weight[i][0][k] * input[idx];
                }
            }
            conv_out[i][j] = relu(conv_out[i][j]);
        }
    }

    // Max pooling layer
    float pool_out[CONV_OUT_CHANNELS][7]; // Result after pooling
    for (int i = 0; i < CONV_OUT_CHANNELS; i++) {
        for (int j = 0; j < 7; j++) {
            float max_val = conv_out[i][2 * j];
            if (conv_out[i][2 * j + 1] > max_val) {
                max_val = conv_out[i][2 * j + 1];
            }
            pool_out[i][j] = max_val;
        }
    }

    // Flatten the pooling layer output
    float fc1_input[CONV_OUT_CHANNELS * 7];
    for (int i = 0; i < CONV_OUT_CHANNELS; i++) {
        for (int j = 0; j < 7; j++) {
            fc1_input[i * 7 + j] = pool_out[i][j];
        }
    }

    // Fully connected layer 1
    float fc1_out[FC1_SIZE];
    for (int i = 0; i < FC1_SIZE; i++) {
        fc1_out[i] = fc1_bias[i];
        for (int j = 0; j < CONV_OUT_CHANNELS * 7; j++) {
            fc1_out[i] += fc1_weight[i][j] * fc1_input[j];
        }
        fc1_out[i] = relu(fc1_out[i]);
    }

    // Fully connected layer 2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = fc2_bias[i];
        for (int j = 0; j < FC1_SIZE; j++) {
            output[i] += fc2_weight[i][j] * fc1_out[j];
        }
    }

    // Apply softmax
    softmax(output, OUTPUT_SIZE, output);
}

int main() {
    // Example input data (replace with actual input)
    float input[INPUT_SIZE] = {-0.5928, 0.1287, -0.2418, -1.6263, -1.8837, 4.9959, -0.5070, -0.2457, 1.0959, -0.4563, -0.0975, -0.1521, -0.4524, -0.1287, -0.1872};
    float output[OUTPUT_SIZE];

    // Perform forward propagation
    forward(input, output);

    printf("Output: NotFalling: %f, Falling: %f\n", output[0], output[1]); // Should be NotFalling: 0.000000, Falling: 1.000000
    return 0;
}