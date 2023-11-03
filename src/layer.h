#ifndef _LAYER_H
#define _LAYER_H

#include <cstring>
#include <time.h>
#include <iostream>
#include <math.h>

typedef enum {
    SIGMOID, 
    RELU, 
    LEAKY
} ACTIVATION;

class Layer {
    public:
        ACTIVATION activate;
        int num_input_nodes;
        int num_nodes;
        float** weights;
        float* outputs;
        float* delta;

        Layer(int num_nodes, int num_input_nodes, ACTIVATION activate=SIGMOID);
        Layer(Layer &layer);
        ~Layer();

        void forward(float *inputs);
        void backward(float *prev_outputs, float *prev_delta, float lr);
        void init();
    private:
        inline float act_func(float x, ACTIVATION act);
        inline float gradient(float x, ACTIVATION activate);
};

inline Layer::Layer(int num_nodes, int num_input_nodes, ACTIVATION activate) 
: num_nodes(num_nodes), num_input_nodes(num_input_nodes), activate(activate) {
    this->weights = new float*[this->num_nodes];
    this->outputs = new float[this->num_nodes];
    this->delta = new float[this->num_nodes];
    this->init();
}


inline Layer::Layer(Layer& layer) 
: num_nodes(layer.num_nodes), num_input_nodes(layer.num_input_nodes), activate(layer.activate) {
    int size = num_nodes * sizeof(float);
    std::memcpy(outputs, layer.outputs, size);
    std::memcpy(delta, layer.delta, size);
    for (int i = 0; i < num_nodes; i++) {
        std::memcpy(weights[i], layer.weights[i], layer.num_input_nodes + 1);
    }
}

inline Layer::~Layer() {
    for (int i = 0; i < num_nodes; i++) {
        delete []weights[i];
    }

    delete []weights;
    delete []outputs;
    delete []delta;
}

inline void Layer::init() {
    memset(outputs, 0, num_nodes * sizeof(float));
    memset(delta, 0, num_nodes * sizeof(float));
    srand(time(0));

    for (int i = 0; i < num_nodes; i++) {
        float *curw = new float[num_input_nodes + 1];
        weights[i] = curw;
        for (int w = 0; w < num_input_nodes + 1; w++) {
            curw[w] = rand() % 1000 * 0.001 - 0.5;
        }
    }
}

inline float Layer::act_func(float x, ACTIVATION act) {
    switch (act) {
        case SIGMOID:
            return (1.0 / (1.0 + exp(-x)));
        case RELU:
            return x * (x > 0);
        case LEAKY:
            return (x > 0) ? x : 0.1 * x;
        default:
            return x;
    }
}

// gradient of activation function
inline float Layer::gradient(float x, ACTIVATION act) {
    switch (act) {
        case SIGMOID:
            return x * (1.0 * x);
        case RELU:
            return (x > 0);
        case LEAKY:
            return (x > 0) ? 1 : 0.1;
        default:
            return 1.0;
    }
}

inline void Layer::forward(float *inputs) {
    for (int i = 0; i < num_input_nodes; i++) {
        float* curw = weights[i];
        float x = 0;
        int k = 0;
        for (; k < num_input_nodes; ++k) {
            x += curw[k] * inputs[k];
        }
        x += curw[k];
        outputs[i] = act_func(x, activate);
    }
}

inline void Layer::backward(float *prev_outputs, float *prev_delta, float lr) {
    for (int i = 0; i < num_nodes; i++) {
        float* curw = weights[i];
        // get gradient
        float delta = this->delta[i] * gradient(outputs[i], activate);
        int k = 0;
        for (; k < num_input_nodes; k++) {
            if (prev_delta != nullptr) {
                prev_delta[k] += curw[k] * delta;
            }
            // update weights
            curw[k] += delta * lr * prev_outputs[k];
        }
        // update bias
        curw[i] += delta * lr;
    }
}


#endif