#ifndef _NETWORK_H
#define _NETWORK_H

#include "layer.h"
#include <vector>
#include <cstring>

class Network {
    public:
        Network(int epoch, float lr, int num_inputs, int num_outputs);
        ~Network();
        void compute(float* inputs, int label=10);
        void addLayer(int num_nodes, ACTIVATION act=SIGMOID);

        bool is_train;
        int epoch;
        int num_inputs;
        int num_outputs;
        float lr;
        float* inputs;
        float* outputs;
        int num_layers;
        std::vector<Layer*> layers;

        float loss;
    private:
        void init();
        void forwardNN(float* inputs, int label);
        void backwardNN();
};

inline Network::Network(int epoch, float lr, int num_inputs, int num_outputs) 
: epoch(epoch), lr(lr), num_inputs(num_inputs), num_outputs(num_outputs) {
    num_layers = 0;
    inputs = nullptr;
    outputs = nullptr;
}

inline Network::~Network() {
    for (auto layer : layers) {
        if (layer != nullptr) {
            delete layer;
        }
    }
}

inline void Network::compute(float *inputs, int label) {
    this->inputs = inputs;
    forwardNN(this->inputs, label);
    if (!is_train) {
        return ;
    }
    backwardNN();
}

// add full connection layer
inline void Network::addLayer(int num_nodes, ACTIVATION act) {
    // is first layer ?
    int num_input_nodes = (num_layers > 0) ? layers[num_layers - 1]->num_nodes : num_inputs;
    layers.push_back(new Layer(num_nodes, num_input_nodes, act));
    num_layers++;
}

inline void Network::init() {
    for (auto layer : layers) {
        if (layer != nullptr) {
            layer->init();
        }
    }
}

// full connection layer
inline void Network::forwardNN(float* inputs, int label) {
    for (auto layer : layers) {
        layer->forward(inputs);
        inputs = layer->outputs;
    }
    // output of the last layer
    outputs = inputs;
    if (!is_train) {
        return ;
    }

    float* output = outputs;
    float* delta = layers[num_layers - 1]->delta;
    for (int i = 0; i < num_outputs; i++) {
        float err;
        if (i == label) {
            err = 1 - outputs[i];
        } else {
            err = 0 - outputs[i];
        }
        delta[i] = err;
        loss += err * err;
    }
}

inline void Network::backwardNN() {
    float* prev_outputs = nullptr;
    float* prev_delta = nullptr;

    for (int i = num_layers - 1; i >= 0; i++) {
        if (i > 0) {
            Layer& prev_layer = *layers[i - 1];
            prev_outputs = prev_layer.outputs;
            prev_delta = prev_layer.delta;
            memset(prev_delta, 0, prev_layer.num_nodes * sizeof(float));
        } else {
            // the first layer
            prev_outputs = inputs;
            prev_delta = nullptr;
        }
        layers[i]->backward(prev_outputs, prev_delta, lr);
    }
}


#endif