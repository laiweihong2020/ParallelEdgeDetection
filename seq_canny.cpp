// CPP program to perform Canny Edge Detection on Images
#include <fstream>
#include <iostream>
#include <string>
#include <bits/stdc++>

# define WIDTH  512
#define HEIGHT  512

using namespace std;

int **new2d (int width, int height) {
    int **dp = new int *[width];
    size_t size = width;
    size *= height;
    int *dp0 = new int[size];
    if (!dp || !dp0) {
        std::cerr << "Failed to create 2d array" << std::endl;
        exit(1);
    }
    dp[0] = dp0;
    for(int i = 1; i< width; i++) {
        dp[i] = dp[i-1] + height;
    }
    return dp;
}

void showMatrix(int **mat, int w, int h) {
    for(int i = 0; i < w; i++) {
        for(int j = 0; j < h; j++) {
            printf("%d", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    // Create the 2d matrix at which the image is stored
    int **imageMat = new2d(WIDTH, HEIGHT);
    size_t size = WIDTH;
    size *= HEIGHT;
    memset(imageMat[0], 0, size);

    ifstream infile("image_matrices/lena512.txt");
    string line;

    int i = 0;
    int j = 0;

    while(getline(infile, line))
    {
        // Split the string based on spaces and fill the matrix
        stringstream ss(line);
        int token;
        while (ss >> token) {
            // WErite the token into the matrix
            imageMat[i][j] = token;
            i++;
        }
        j++;
    }

    showMatrix(imageMat, WIDTH, HEIGHT);
    infile.close();
}