#include "layer.h"
#include "layer.h"
#include "network.h"

int main() {
    int network_inputs = 28 * 28;
    int network_outputs = 10;
    int epoches = 10;
    float lr = 0.1;

    Network *network = new Network(epoches, lr, network_inputs, network_outputs);
    network->addLayer(256, SIGMOID);
    network->addLayer(128, SIGMOID);
    network->addLayer(network->num_outputs, SIGMOID);
    network->is_train = false;

    return 0;
}

