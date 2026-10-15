#ifndef IMAGE_IO_H
#define IMAGE_IO_H

bool loadGrayscaleImage(const char* filename, unsigned char*& data, int& width, int& height);
bool saveGrayscaleImage(const char* filename, const unsigned char* data, int width, int height);
void freeImage(unsigned char* data);

#endif