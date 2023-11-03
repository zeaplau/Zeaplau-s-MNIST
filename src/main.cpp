#include "test.h"
#include "layer.h"
#include <iostream>

int main() {
    Layer* l = new Layer(100, 100);
    std::cout << l->num_nodes << std::endl;
    return 0;
}

