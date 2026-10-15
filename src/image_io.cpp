#include "image_io.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool loadGrayscaleImage(const char* filename, unsigned char*& data, int& width, int& height) {
    int channels;
    data = stbi_load(filename, &width, &height, &channels, 1); // fuerza escala de grises
    if (!data) {
        std::cerr << "Error loading image: " << filename << std::endl;
        return false;
    }
    return true;
}

bool saveGrayscaleImage(const char* filename, const unsigned char* data, int width, int height) {
    int success = stbi_write_png(filename, width, height, 1, data, width);
    if (!success) {
        std::cerr << "Error saving image: " << filename << std::endl;
        return false;
    }
    return true;
}

void freeImage(unsigned char* data) {
    stbi_image_free(data);
}