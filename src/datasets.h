#ifndef __DATASETS_H__
#define __DATASETS_H__

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

struct image_dataset {
	uint32_t n_images;
	uint32_t n_rows;
	uint32_t n_cols;
	uint8_t *images;
};

struct label_dataset {
	uint32_t n_items;
	uint8_t *labels;
};

void display_image(struct image_dataset *dataset, uint32_t index); 
void free_images(struct image_dataset *dataset);
struct image_dataset read_images(const char *path);
void free_labels(struct label_dataset *dataset);
struct label_dataset read_labels(const char *path);

#endif // __DATASETS_H__
