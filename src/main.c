#include "datasets.h"
#include "net.h"

#define TRAIN_SPLIT 0.6
#define BATCH_SIZE 128
#define N_PIXELS 784 
#define N_EPOCHS 1
#define LEARNING_RATE 0.001f

int main(int argc, char *argv[]) {
	(void) argc;
	(void) argv;

	struct network network;
	struct label_dataset labels = read_labels("../t10k-labels-idx1-ubyte");
	struct image_dataset images = read_images("../t10k-images-idx3-ubyte");

	uint32_t train_size = (uint32_t) (images.n_images * TRAIN_SPLIT);
	uint32_t test_size = images.n_images - train_size;

#ifdef DEBUG
	fprintf(stderr, "[*] Number of rows: %u\n", images.n_rows);
	fprintf(stderr, "[*] Number of columns: %u\n", images.n_cols);
	
	fprintf(stderr, "[*] Training on %u samples\n", train_size);
	fprintf(stderr, "[*] Testing on %u samples\n", test_size);
#endif

	init_layer(&network.hidden, N_PIXELS, 64);
	init_layer(&network.output, 64, 10);

	float image_buffer[N_PIXELS] = {0.f};

	for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
		float total_loss = 0.f;
		
		for (uint32_t batch = 0; batch < train_size; batch += BATCH_SIZE) {

			for (uint32_t item_index = 0; item_index < BATCH_SIZE && (batch + item_index) < train_size; item_index++) {
				uint32_t global_index = batch + item_index;

				// Normalize the data
				for (uint32_t pixel_index = 0; pixel_index < N_PIXELS; pixel_index++) {
					image_buffer[pixel_index] = images.images[global_index * N_PIXELS + pixel_index] / 255.0f;
				}

				train(&network, image_buffer, labels.labels[global_index], LEARNING_RATE);
				memset(network.hidden.output, 0, network.hidden.output_size * sizeof(float));
				memset(network.output.output, 0, network.output.output_size * sizeof(float));
				forward_pass(&(network.hidden), image_buffer);
				apply_relu(&(network.hidden));

				forward_pass(&(network.output), network.hidden.output);
				apply_softmax(&(network.output));

				total_loss += -logf(network.output.output[labels.labels[global_index]] + 1e-10f);
			}
		}

		uint32_t correct_guesses = 0;
		for (uint32_t image_index = train_size; image_index < images.n_images; image_index++) {
			for (uint32_t pixel_index = 0; pixel_index < N_PIXELS; pixel_index++) {
				image_buffer[pixel_index] = images.images[image_index * N_PIXELS + pixel_index] / 255.0f;
			}
			
			uint32_t predicted_label = predict(&network, image_buffer);
			if (predicted_label == labels.labels[image_index]) {
				correct_guesses++;
			}
		}

		fprintf(stderr, "%u %u\n", correct_guesses, images.n_images);

		fprintf(stderr, "Epoch %d, Accuracy %.2f%%, Avg Loss %.4f\n", epoch + 1, (float) correct_guesses / test_size * 100, total_loss / train_size);
	}

	free_layer(&(network.hidden));
	free_layer(&(network.output));

	free_labels(&labels);
	free_images(&images);

	return 0;
}
