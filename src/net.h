#ifndef __NET_H__
#define __NET_H__

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

struct layer {
	float *weights;
	float *biases;

	uint32_t input_size;
	uint32_t output_size;

	float *output;
	float *gradient;
};

struct network {
	struct layer hidden;
	struct layer output;
};


void init_layer(struct layer *layer, uint32_t input_size, uint32_t output_size);
void free_layer(struct layer *layer);
void forward_pass(struct layer *layer, float *input);
void backward_pass(struct layer *layer, float *input, float *input_grad, float *output_grad, float learning_rate);
void apply_relu(struct layer *layer);
void apply_softmax(struct layer *layer);
void train(struct network *network, float *input, uint32_t label, float learning_rate);
uint8_t predict(struct network *network, float *input);

#endif // __NET_H__
