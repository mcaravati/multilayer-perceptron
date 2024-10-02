#include "datasets.h"

uint32_t flip_bytes(uint32_t value) {
	uint32_t result = 0;

	result |= (value & 0xff000000) >> 24;
	result |= (value & 0x00ff0000) >> 8;
	result |= (value & 0x0000ff00) << 8;
	result |= (value & 0x000000ff) << 24;

	return result;
}

void display_image(struct image_dataset *dataset, uint32_t index) {
	uint32_t offset = index * dataset->n_cols * dataset->n_rows;

	for (uint32_t pixel_index = 0; pixel_index < (dataset->n_cols * dataset->n_rows); pixel_index++) {
		if (pixel_index % dataset->n_cols == 0) {
			fprintf(stdout, "\n");
		}

		fprintf(stdout, *(dataset->images + offset + pixel_index) > 250 ? "X" : " ");
	}
}

void free_images(struct image_dataset *dataset) {
	free(dataset->images);
}

struct image_dataset read_images(const char *path) {
	struct image_dataset loaded_dataset;
	uint32_t buffer = 0;
	FILE *images_file = fopen(path, "rb");

	if (! images_file) {
		fprintf(stderr, "[-] Could not open image dataset: %s\n", path);
		loaded_dataset.n_images = -1;
		return loaded_dataset;
	}

	// Read the magic number, even if we won't actively use it
	fread(&buffer, 4, 1, images_file);

	// Read number of images in the dataset
	fread(&buffer, 4, 1, images_file);
	loaded_dataset.n_images = flip_bytes(buffer);

	// Read number of rows in each image
	fread(&buffer, 4, 1, images_file);
	loaded_dataset.n_rows = flip_bytes(buffer);
	
	// Read number of columns in each image
	fread(&buffer, 4, 1, images_file);
	loaded_dataset.n_cols = flip_bytes(buffer);

	loaded_dataset.images = calloc(loaded_dataset.n_images, loaded_dataset.n_rows * loaded_dataset.n_cols);
	if (! loaded_dataset.images) {
		fprintf(stderr, "[-] Could not allocate enough space for %u images\n", loaded_dataset.n_images);
		return loaded_dataset;
	}	

	fread(loaded_dataset.images, loaded_dataset.n_rows * loaded_dataset.n_cols, loaded_dataset.n_images, images_file);

	if (fclose(images_file)) {
		fprintf(stderr, "[-] Error while closing images dataset\n");
	}

	return loaded_dataset;
}

void free_labels(struct label_dataset *dataset) {
	free(dataset->labels);
}

struct label_dataset read_labels(const char *path) {
	struct label_dataset loaded_dataset;
	uint32_t buffer = 0;
	FILE *labels_file = fopen(path, "rb");
	
	if (! labels_file){
		fprintf(stderr, "[-] Could not open label dataset: %s\n", path);
		loaded_dataset.n_items = -1;
		return loaded_dataset;
	}

	// Read the magic number, even if we won't actively use it
	fread(&buffer, 4, 1, labels_file);

	// Read number of labels
	fread(&buffer, 4, 1, labels_file);
	loaded_dataset.n_items = flip_bytes(buffer);

	// Allocate space for labels array
	loaded_dataset.labels = calloc(loaded_dataset.n_items, sizeof(unsigned int));

	if (! loaded_dataset.labels) {
		fprintf(stderr, "[-] Could not allocate enough space for %u labels\n", loaded_dataset.n_items);
		return loaded_dataset;
	}

	fread(loaded_dataset.labels, 1, loaded_dataset.n_items, labels_file);

	if (fclose(labels_file)) {
		fprintf(stderr, "[-] Error while closing labels dataset\n");
	}

	return loaded_dataset;
}
