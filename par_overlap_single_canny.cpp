// CPP program to perform Canny Edge Detection on Images
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <cmath>
#include <mpi.h>
#include <queue>

# define WIDTH          5000
# define HEIGHT         5000
# define KERNEL_SIZE    5
# define HIGH_THRESHOLD 0.2
# define LOW_THRESHOLD  0.05

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

void writeToFile(string filePath, double **mat, int width, int height) {
    ofstream outfile(filePath);
    
    for(int j = 0; j < height; ++j) {
        string s = "";
        for(int i = 0; i < width; ++i) {
            double val = mat[i][j];
            string valInt = to_string((int)round(val));

            s += valInt;
            s += " ";
        }
        outfile << s << "\n";
    }
}

// Currently only works for kernel of size 5x5
void **gaussianBlur(double **im, double **kernel, double **results, int width, int height) {
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

void doubleThresholding(double **im, int max) {
    if (max > 255) {
        max = 255;
    }

    double highThreshold = max * HIGH_THRESHOLD;
    double lowThreshold = highThreshold * LOW_THRESHOLD;

    for(int x = 0; x < WIDTH; ++x) {
        for(int y = 0; y < HEIGHT; ++y) {
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
void hysteresis(double **im, int width, int height, int rank) {
    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            // See if the weak component is connected to a strong component
            if(im[x][y] == 25) {
                // Don't process edges for now
                if(x == 0 || y == 0 || x == width-1 || x == height-1) {
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

    // Process edge results that are important for other nodes
    // Needs to be refactored
    if(rank == 1) {
        queue<int> leftMostQueue;
        for(int i = 0; i < height; ++i) {
            double pixel_value = im[width-1][i];
            if(pixel_value == 255) {
                leftMostQueue.push(i);
            }
        }
        // Convert the queue into an array
        int *leftMost = new int[leftMostQueue.size()];
        for(int i = 0; i < leftMostQueue.size(); ++i) {
            leftMost[i] = leftMostQueue.front();
            leftMostQueue.pop();
        }

        // Send the results to the other components
        MPI_Ssend(leftMostQueue.size(), 1, MPI_INT, 0, 1, MPI_COMM_WORLD)
        MPI_Bsend(leftMost, leftMostQueue.size(), MPI_INT, 0, 2, MPI_COMM_WORLD)
    } else if(rank == 2) {
        queue<int> topMostQueue;
        for(int i = 0; i < width; ++i) {
            double pixel_value = im[0][i];
            if(pixel_value == 255) {
                topMostQueue.push(i);
            }
        }

        // Convert the queue into an array
        int *topMost = new int[topMostQueue.size()];
        for(int i = 0; i < topMostQueue.size(); ++i) {
            topMost[i] = topMostQueue.front();
            topMostQueue.pop();
        }

        // Send the results to the other components
        MPI_Ssend(topMostQueue.size(), 1, MPI_INT, 0, 1, MPI_COMM_WORLD)
        MPI_Bsend(topMost, topMostQueue.size(), MPI_INT, 0, 2, MPI_COMM_WORLD)
    } else if (rank == 3) {
        queue<int> leftMostQueue;
        for(int i = 0; i < height; ++i) {
            double pixel_value = im[width-1][i];
            if(pixel_value == 255) {
                leftMostQueue.push(i);
            }
        }
        // Convert the queue into an array
        int *leftMost = new int[leftMostQueue.size()];
        for(int i = 0; i < leftMostQueue.size(); ++i) {
            leftMost[i] = leftMostQueue.front();
            leftMostQueue.pop();
        }

        // Send the results to the other components
        MPI_Ssend(leftMostQueue.size(), 1, MPI_INT, 2, 1, MPI_COMM_WORLD)
        MPI_Bsend(leftMost, leftMostQueue.size(), MPI_INT, 2, 2, MPI_COMM_WORLD)

        queue<int> topMostQueue;
        for(int i = 0; i < width; ++i) {
            double pixel_value = im[0][i];
            if(pixel_value == 255) {
                topMostQueue.push(i);
            }
        }

        // Convert the queue into an array
        int *topMost = new int[topMostQueue.size()];
        for(int i = 0; i < topMostQueue.size(); ++i) {
            topMost[i] = topMostQueue.front();
            topMostQueue.pop();
        }

        // Send the results to the other components
        MPI_Ssend(topMostQueue.size(), 1, MPI_INT, 1, 1, MPI_COMM_WORLD)
        MPI_Bsend(topMost, topMostQueue.size(), MPI_INT, 1, 2, MPI_COMM_WORLD)
    }

    if(rank == 0) {
        // Wait for process 1 and 2
        // Get the size of the array
        // Allocate the memory 
        // Process and free the memory
        int leftMostSize = 0;
        MPI_Recv(&leftMostSize, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);

        // Allocate memory for the message
        int *leftMost = new int[leftMostSize];
        MPI_Recv(leftMost, leftMostSize, MPI_INT, 1, 2, MPI_COMM_WORLD);

        // Process the array
        // Repeat for rank 2
    } else if(rank == 1) {
        // Wait for rank 3
    } else if(rank == 2) {
        // Wait for rank 3
    }
}

void processImage(int rank, int width, int height) {
    // Create the 2d matrix at which the image is stored
    double **imageMat = new2d(width+4, height+4);
    size_t size = width+4;
    size *= height+4;
    std::memset(imageMat[0], 0, size);

    // Read the split file
    string filePath = "image_matrices/overlap_chunks/" + to_string(rank);
    ifstream infile(filePath);
    string line;

    int i = 0;
    int j = 0;

    while(getline(infile, line)) {
        // Split the string based on spaces and fill the matrix
        stringstream ss(line);
        int token;
        while(ss >> token) {
            imageMat[i][j] = token;
            i++;
        }
        i = 0;
        j++;
    }

    // We should have an image matrix here
    // Apply noise reducion using a Gaussian kernel
    // Create the Gaussian kernel
    double **gaussianKernel = new2d(KERNEL_SIZE, KERNEL_SIZE);
    size_t k_size = KERNEL_SIZE;
    k_size *= KERNEL_SIZE;
    std::memset(gaussianKernel[0], 0, k_size);

    // Generate the Gaussian kernel
    gaussianKernelGeneration(gaussianKernel, 1, KERNEL_SIZE);

    // Apply the convolution operation (may be worth parallelising)
    double **blurredImageMat = new2d(width, height);
    size_t bl_size = width;
    bl_size *= height;
    std::memset(blurredImageMat[0], 0, bl_size);

    gaussianBlur(imageMat, gaussianKernel, blurredImageMat, width, height);

    delete[] imageMat;
    delete[] gaussianKernel;

    // Gradient Calculation
    // Create the 2d matrix at which the image is stored
    double **G = new2d(width, height);
    size = width;
    size *= height;
    std::memset(G[0], 0, size);

    // Create the 2d matrix at which the image is stored
    double **theta = new2d(width, height);
    std::memset(theta[0], 0, size);

    gradientCalculation(blurredImageMat, G, theta, width, height);

    delete[] blurredImageMat;
    // End of gradient calculation

    // Non-Maximum Suppression
    double **nonMaxSuppress = new2d(width, height);
    std::memset(nonMaxSuppress[0], 0, size);
    double max = 0;

    max = nonMaxSuppression(G, theta, nonMaxSuppress, width, height);

    delete[] G;
    delete[] theta;

    // End of non-maximum suppression

    // Double thresholding
    doubleThresholding(nonMaxSuppress, max);
    // End of double thresholding

    // Edge tracking by hysteresis
    hysteresis(nonMaxSuppress, width, height);
    // End of edge tracking by hysteresis

    infile.close();
}

int main(int argc, char **argv) {
    // Separate the image into chunks
    
    // Create the 2d matrix at which the image is stored
    double **imageMat = new2d(WIDTH, HEIGHT);
    size_t size = WIDTH;
    size *= HEIGHT;
    std::memset(imageMat[0], 0, size);

    ifstream infile("image_matrices/lena.txt");
    string line;

    int i = 0;
    int j = 0;

    while(getline(infile, line)) {
        // Split the string based on spaces and fill the matrix
        stringstream ss(line);
        int token;
        while(ss >> token) {
            imageMat[i][j] = token;
            i++;
        }
        i = 0;
        j++;
    }

    // The image should be loaded into memory now
    // Start dividing the image into 4 chunks
    int dx = WIDTH/2;
    int dy = HEIGHT/2;

    double **tempImageMat = new2d(dx+4, dy+4);
    size_t t_size = dx+4;
    t_size *= dy+4;
    std::memset(tempImageMat[0], 0, t_size);

    for(int x = 0; x < 2; x++) {
        for(int y = 0; y < 2; y++) {
            for(int xx = 0; xx < dx+4; xx++) {
                for (int yy = 0; yy < dy+4; yy++)
                {
                    if(x*dx + xx > WIDTH-1 || y*dy + yy > HEIGHT-1) {
                        // This is just padding done in real time
                        tempImageMat[xx][yy] = 0;
                    } else {
                        tempImageMat[xx][yy] = imageMat[x*dx + xx][y*dy + yy];
                    }
                }   
            }
            // Write the chunk to file
            string s = "image_matrices/overlap_chunks/" + to_string(x*2 + y); 
            writeToFile(s, tempImageMat, dx+4, dy+4);
        }
    }

    int rank, world_size;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    processImage(rank, dx, dy, rank);

    MPI_Finalize();
}