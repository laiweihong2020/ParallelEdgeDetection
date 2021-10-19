#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <cmath>
#include <mpi.h>
#include <queue>

#define HIGH_THRESHOLD 0.2
#define LOW_THRESHOLD 0.05
#define KERNEL_SIZE    5

using namespace std;

double **new2d (int width, int height) {
    double **dp = new double *[width];
    size_t size = width;
    size *= height;
    double *dp0 = new double[size];
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

void gaussianKernelGeneration(double **mat, int sigma, int size) {
    // Initialise the standard deviation to sigma
    double r, s = 2.0 * sigma * sigma;

    // sum for normalisation
    double sum = 0.0;
    int k = (size - 1)/2;

    // generate the kernel based on predefined size
    for (int x = -k; x <= k; ++x) {
        for (int y = -k; y <= k; ++y) {
            r = sqrt(x * x + y * y);
            mat[x + 2][y + 2] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += mat[x + 2][y + 2];
        }
    } 

    // normalise the kernel
    for(int i = 0; i < 5; ++i) {
        for(int j = 0; j < 5; ++j) {
            mat[i][j] /= sum;
        }
    }
}

void gaussianBlur(double **im, double **kernel, double **results, int width, int height) {
    double **convolutedIm = new2d(KERNEL_SIZE, KERNEL_SIZE);
    size_t size = KERNEL_SIZE;
    size *= KERNEL_SIZE;
    std::memset(convolutedIm[0], 0, size);

    double **result = new2d(KERNEL_SIZE, KERNEL_SIZE);
    size_t k_size = KERNEL_SIZE;
    k_size *= KERNEL_SIZE;
    std::memset(result[0], 0, k_size);

    int k = (KERNEL_SIZE - 1)/2;

    // Iterate through every pixel in the image
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            // Crop the pixel region to apply gaussian blur
            for (int xx = x - k; xx <= x + k; ++xx){
                for(int yy = y - k; yy <= y+k; ++yy) {
                    if(xx < 0 || yy < 0 || xx > width-1 || yy > height-1) {
                        // This is just padding done in real time
                        convolutedIm[xx + k - x][yy + k -y] = 0;
                    } else {
                        convolutedIm[xx + k - x][yy + k -y] = im[xx][yy];
                    }
                }
            }

            // Here we should have a local 5x5 region of our picture
            // Now we apply matrix multiplication onto the cropped image
            for(int i = 0; i < KERNEL_SIZE; ++i) {
                for(int j = 0; j < KERNEL_SIZE; ++j) {
                    result[i][j] += convolutedIm[i][j] * kernel[i][j];
                }
            }

            // Add up all the values and set that as the new intensity
            for(int i = 0; i < KERNEL_SIZE; ++i) {
                for(int j = 0; j < KERNEL_SIZE; ++j) {
                    results[x][y] += result[i][j];
                    result[i][j] = 0;
                }
            }
            // // results should start to be populated at this point
        }
    }

    // Free resources after calculation is completed
    delete[] result;
    delete[] convolutedIm;
}

void gradientCalculation(double **im, double **G, double **theta, int width, int height) {
    int Kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Ky[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    double **convolutedIm = new2d(3, 3);
    size_t size = 3;
    size *= 3;
    std::memset(convolutedIm[0], 0, size);

    // Convolve the filter onto the image
    // Iterate through every pixel in the image
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            int i_x = 0;
            int i_y = 0;
            // Crop the pixel region to apply gaussian blur
            for (int xx = x - 1; xx <= x + 1; ++xx){
                for(int yy = y - 1; yy <= y+1; ++yy) {
                    if(xx < 0 || yy < 0 || xx > width-1 || yy > height-1) {
                        // This is just padding done in real time
                        convolutedIm[xx + 1 - x][yy + 1 -y] = 0;
                    } else {
                        convolutedIm[xx + 1 - x][yy + 1 -y] = im[xx][yy];
                    }
                }
            }

            // Here we should have the convolved im
            // Now we apply matrix multiplication onto the cropped image
            for(int i = 0; i < 3; ++i) {
                for(int j = 0; j < 3; ++j) {
                    i_x += convolutedIm[i][j] * Kx[i][j];
                    i_y += convolutedIm[i][j] * Ky[i][j];
                }
            }

            // Compute the G value and theta value
            G[x][y] = sqrt(i_x * i_x + i_y * i_y);
            theta[x][y] = atan2 (i_y , i_x);
        }
    }
}

// Make sure to keep track of the maximum value here
int nonMaxSuppression(double **G, double **theta, double **result, int width, int height) {
    double max = 0;
    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            // Get the angle for that location
            double angle = theta[x][y] * 180 / M_PI;

            if(angle < 0) {
                angle += 180;
            }

            int q[2] = {};
            int r[2] = {};

            // Handle cases for angles
            if((0 <= angle < 22.5) || 157.5 <= angle <= 180) {
                q[0] = x;
                q[1] = y+1;
                r[0] = x;
                r[1] = y-1;
            } else if(22.5 <= angle < 67.5) {
                q[0] = x+1;
                q[1] = y-1;
                r[0] = x-1;
                r[1] = y+1;
            } else if(67.5 <= angle < 112.5) {
                q[0] = x+1;
                q[1] = y;
                r[0] = x-1;
                r[1] = y;
            } else if(112.5 <= angle < 157.5) {
                q[0] = x-1;
                q[1] = y-1;
                r[0] = x+1;
                r[1] = y+1;
            }

            // get the pixel values
            double q_val = 255;
            double r_val = 255;
            
            // Check index bounds and make appropriate actions
            if(q[0] < 0 || q[1] < 0 || r[0] < 0 || r[1] < 0 
                || q[0] > width-1 || r[0] > width-1 || q[1] > height-1 || r[1] > height-1) {
                    // The pixel is at the edge and cannot be performed non max suppression
                    continue;
                } else {
                    // q and r values within bounds
                    q_val = G[q[0]][q[1]];
                    r_val = G[r[0]][r[1]];
                }

            // Check if the pixel intensity is greater
            if(G[x][y] > q_val && G[x][y] > r_val) {
                if(max < G[x][y]) {
                    max = G[x][y];
                }
                result[x][y] = G[x][y];
            } else {
                result[x][y] = 0;
            }
        }
    }
    return max;
}

void doubleThresholding(double **im, int max, int width, int height) {
    if (max > 255) {
        max = 255;
    }

    double highThreshold = max * HIGH_THRESHOLD;
    double lowThreshold = highThreshold * LOW_THRESHOLD;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            if(im[x][y] > highThreshold) {
                im[x][y] = 255;
            } else if(im[x][y] > lowThreshold) {
                im[x][y] = 25;
            } else {
                im[x][y] = 0;
            }
        }
    }
}

// Edge tracking
// Communication between parallel components
void hysteresis(double **im, int width, int height) {
    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            // See if the weak component is connected to a strong component
            if(im[x][y] == 25) {
                // Don't process edges for now
                if(x == 0 || y == 0 || x == WIDTH-1) {
                    continue;
                }

                // Check surrounding blocks to see if there are any strong points
                if(im[x+1][y] == 255 || im[x+1][y+1] == 255 || im[x][y+1] == 255 || im[x-1][y+1] == 255
                    || im[x-1][y] == 255 || im[x-1][y-1] == 255 || im[x][y-1] == 255 || im[x+1][y-1] == 255) {
                        im[x][y] = 255;
                    }
                else {
                    im[x][y] = 0;
                }
            }
        }
    }
}

void processImage(string filePath, int yOffset, int width, int height) {
    // Create a 2d matrix at which the image is stored
    double **imageMat = new2d(width, height+5);
    size_t size = width;
    size *= height+5;
    std::memset(imageMat[0], 0, size);

    // Read the file and load data into the matrix
    ifstream infile(filePath);
    string line;

    int i = 0;
    int j = 0;
    
    while(getline(infile, line)) {
        // Split the string based on spaces and fill the matrix
        stringstream ss(line);
        int token;
        if(j < yOffset) {
            j++;
            continue;
        }

        if(j-yOffset == height+4) {
            // Max height for matrix achieved
            break;
        }

        // j should reach the offset here
        while(ss >> token) {
            imageMat[i][j-yOffset] = token;
            i++;
        }
        i = 0;
        j++;
    }

    // We should have an image matrix here
    // Apply noise reduction using a Gaussian kernel
    // Create the Gaussian kernel
    double **gaussianKernel = new2d(KERNEL_SIZE, KERNEL_SIZE);
    size_t k_size = KERNEL_SIZE;
    k_size *= KERNEL_SIZE;
    std::memset(gaussianKernel[0], 0, k_size);

    // Generate the Gaussian kernel
    gaussianKernelGeneration(gaussianKernel, 1, KERNEL_SIZE);

    // Apply the convolution operation
    double **blurredImageMat = new2d(width, height+1);
    size_t bl_size = width;
    bl_size *= height+1;
    std::memset(blurredImageMat[0], 0, bl_size);

    gaussianBlur(imageMat, gaussianKernel, blurredImageMat, width, height+1);

    delete[] imageMat;
    delete[] gaussianKernel;

    // Gradient Calculation
    // Create the 2d matrix at which the image is stored
    double **G = new2d(width, height+1);
    size = width;
    size *= height + 1;
    std::memset(G[0], 0, size);

    // Create the 2d matrix at which the image is stored
    double theta = new2d(width, height+1);
    std::memset(theta[0], 0, size);

    gradientCalculation(blurredImageMat, G, theta, width, height+1);

    delete[] blurredImageMat;

    double **nonMaxSuppress = new2d(width, height + 1);
    std::memset(nonMaxSuppress[0], 0, size);
    double max = 0;

    max = nonMaxSuppress(G, theta, nonMaxSuppress, width, height + 1);

    delete[] G;
    delete[] theta;

    doubleThresholding(nonMaxSuppress, max, width, height+1);

    hysteresis(nonMaxSuppress, width, height);

    // Save the results by writing into the file
    
}

// We assume that all the processes only have 1GB of memory
int main(int argc, char **argv) {
    
    int rank, world_size;
    int ierr;
    const int overlap = 100;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process in charge of coordinating and distributing work
    // Root will also be in charge of side resolution
    if(rank == 0) {

    }else {
        // Worker process waiting for work
        // Worker shoudl receive a file name and an array with parameters from the root
        // If no work is left, a message will be sent
        char *filePath = new char[100];
        int *param = new int[2];

        MPI_Recv(message, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(param, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // offsetX will represent the x location of the chunk and offset Y will represent the y location of the chunk
        int width = param[0];
        int height = param[1];
        int yOffset = param[2];
        int fileType = param[3]; // 0 for full image, 1 for partial image

        std::string filePathStr = std::string(filePath)

        processImage(filePathStr, yOffset, width, height);
    }
}