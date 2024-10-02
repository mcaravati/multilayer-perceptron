#include "net.h"

// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float randn() {
	float u1 = ((float) rand() / (RAND_MAX));
	float u2 = ((float) rand() / (RAND_MAX));

	float z = sqrtf(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);

	return z;
}

void init_layer(struct layer *layer, uint32_t input_size, uint32_t output_size) {
	layer->input_size = input_size;
	layer->output_size = output_size;
	layer->weights = malloc(input_size * output_size * sizeof(float));
	layer->biases = calloc(output_size, sizeof(float));
	layer->output = calloc(output_size, sizeof(float));
	layer->gradient = calloc(output_size, sizeof(float));

	if((! layer->weights) || (! layer->biases)) {
		fprintf(stderr, "[-] Couldn't allocate enough space for weights or biases");
		return;
	}

	float standard_deviation = sqrtf(2.0f / input_size);
	for (uint32_t i = 0; i < input_size * output_size; i++) {
		layer->weights[i] = randn() * standard_deviation; 
	}
}

void free_layer(struct layer *layer) {
	free(layer->weights);
	layer->weights = NULL;

	free(layer->biases);
	layer->biases = NULL;

	free(layer->output);
	layer->output = NULL;

	free(layer->gradient);
	layer->gradient = NULL;
}

// Verified
void forward_pass(struct layer *layer, float *input) {
	for (uint32_t neuron_index = 0; neuron_index < layer->output_size; neuron_index++) {
		layer->output[neuron_index] = layer->biases[neuron_index];

		for (uint32_t weight_index = 0; weight_index < layer->input_size; weight_index++) {
			layer->output[neuron_index] += input[weight_index] * layer->weights[weight_index * layer->output_size + neuron_index];
		}
	}
}

// Verified
void backward_pass(struct layer *layer, float *input, float *input_grad, float *output_grad, float learning_rate) {
	for(uint32_t neuron_index = 0; neuron_index < layer->output_size; neuron_index++) {
		for (uint32_t input_index = 0; input_index < layer->input_size; input_index++) {
			uint32_t weight_index = input_index * layer->output_size + neuron_index;
			float grad = output_grad[neuron_index] * input[input_index];

			layer->weights[weight_index] -= learning_rate * grad;
			if (input_grad) {
				input_grad[input_index] += output_grad[neuron_index] * layer->weights[weight_index];
			}

			layer->biases[neuron_index] -= learning_rate * output_grad[neuron_index];
		}
	}
}

// Verified
void apply_relu(struct layer *layer) {
	for (uint32_t neuron_index = 0; neuron_index < layer->output_size; neuron_index++) {
		float value = 0.f;
		if (layer->output[neuron_index] > 0) {
			value = layer->output[neuron_index];
		}

		layer->output[neuron_index] = value;
	}
}

void apply_softmax(struct layer *layer) {
	// Find max value
	float max = layer->output[0];
	for (uint32_t value_index = 0; value_index < layer->output_size; value_index++) {
		if (layer->output[value_index] > max) {
			max = layer->output[value_index];
		}
	}

	float sum = 0.f;
	for (uint32_t value_index = 0; value_index < layer->output_size; value_index++) {
		layer->output[value_index] = expf(layer->output[value_index] - max);
		sum += layer->output[value_index];
	}

	for (uint32_t value_index = 0; value_index < layer->output_size; value_index++) {
		layer->output[value_index] /= sum;
	}
}
	
void train(struct network *network, float *input, uint32_t label, float learning_rate) {
	memset(network->hidden.output, 0, network->hidden.output_size * sizeof(float));
	memset(network->output.output, 0, network->output.output_size * sizeof(float));

	memset(network->hidden.gradient, 0, network->hidden.output_size * sizeof(float));
	memset(network->output.gradient, 0, network->output.output_size * sizeof(float));

	forward_pass(&(network->hidden), input);
	apply_relu(&(network->hidden));

	forward_pass(&(network->output), network->hidden.output);
	apply_softmax(&(network->output)); 

	for (uint32_t value_index = 0; value_index < network->output.output_size; value_index++) {
		network->output.gradient[value_index] = network->output.output[value_index] - (value_index == label);
	}

	backward_pass(&(network->output), network->hidden.output, network->hidden.gradient, network->output.gradient, learning_rate);

	// Relu derivative
	for (uint32_t value_index = 0; value_index < network->hidden.output_size; value_index++) {
		uint8_t value = 0;

		if (network->hidden.output[value_index] > 0) {
			value = 1;
		}

		network->hidden.output[value_index] *= value;
	}

	backward_pass(&(network->hidden), input, NULL, network->hidden.gradient, learning_rate);
}

void display_array(float *array, uint32_t size) {
	for (uint32_t index = 0; index < size; index++) {
		fprintf(stderr, "%.3f ", array[index]);
	}

	fprintf(stderr, "\n");
}

uint8_t predict(struct network *network, float *input) {
	memset(network->hidden.output, 0, network->hidden.output_size * sizeof(float));
	memset(network->output.output, 0, network->output.output_size * sizeof(float));

	forward_pass(&(network->hidden), input);
	// display_array(network->hidden.output, network->hidden.output_size);
	apply_relu(&(network->hidden));

	forward_pass(&(network->output), network->hidden.output);
	// display_array(network->hidden.output, network->hidden.output_size);
	apply_softmax(&(network->output));

	// display_array(network->output.output, network->output.output_size);
	uint8_t max_index = 0;
	for (uint8_t neuron_index = 0; neuron_index < (uint8_t) network->output.output_size; neuron_index++) {
		if (network->output.output[neuron_index] > network->output.output[max_index]) {
			max_index = neuron_index;
		}
	}

	return max_index;
}
