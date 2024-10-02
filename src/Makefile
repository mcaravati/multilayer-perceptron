CC=gcc
CFLAGS=-Wall -Wextra -Werror -g -DDEBUG
LDFLAGS=-lm
BUILD_DIR="../build"

all: create_build_dir build
	$(BUILD_DIR)/main

debug: create_build_dir build
	valgrind -s --leak-check=full --show-leak-kinds=all $(BUILD_DIR)/main

create_build_dir:
	mkdir -p $(BUILD_DIR)

%.o: %.c
	$(CC) $(CFLAGS) -c $< $(LDFLAGS)

build: main.c datasets.o net.o
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/main $^ $(LDFLAGS)
