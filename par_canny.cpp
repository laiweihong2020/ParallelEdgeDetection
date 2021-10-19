#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <cmath>
#include <mpi.h>
#include <queue>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <stdio.h>

#define HIGH_THRESHOLD 0.2
#define LOW_THRESHOLD 0.05
#define KERNEL_SIZE    50

using namespace std;

/**
* Utility function to create 2d arrays
*/
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
            mat[x + k][y + k] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += mat[x + k][y + k];
        }
    } 

    // normalise the kernel
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
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
                if(x == 0 || y == 0 || x == width-1) {
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

void processImage(string fileName, int yOffset, int width, int height, int fileType, int messageTag) {
    // Create a 2d matrix at which the image is stored
    double **imageMat = new2d(width, height+5);
    size_t size = width;
    size *= height+5;
    std::memset(imageMat[0], 0, size);

    // Read the file and load data into the matrix
    string filePath = "image_matrices/" + fileName;
    ifstream infile(filePath);
    string line;

    int i = 0;
    int j = 0;
    cout << "yOffset: " << yOffset << endl;
    
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
    double **theta = new2d(width, height+1);
    std::memset(theta[0], 0, size);

    gradientCalculation(blurredImageMat, G, theta, width, height+1);

    delete[] blurredImageMat;

    double **nonMaxSuppress = new2d(width, height + 1);
    std::memset(nonMaxSuppress[0], 0, size);
    double max = 0;

    max = nonMaxSuppression(G, theta, nonMaxSuppress, width, height + 1);

    delete[] G;
    delete[] theta;

    doubleThresholding(nonMaxSuppress, max, width, height+1);

    hysteresis(nonMaxSuppress, width, height);

    // Save the results by writing into the file
    if(fileType == 0) {
        // This means that the entire image is loaded
        filePath = "image_matrices/results/" + fileName;
        writeToFile(filePath, nonMaxSuppress, width, height);
    } else if(fileType == 1) {
        // This means that this is a partial image
        // Send a message to the root process to get the filename
        // The root process will resolve the combination process
        char *outFileName = new char[100];
        int statusResponse = 1;
            cout << "image Processing completed" << endl;
        MPI_Ssend(&statusResponse, 1, MPI_INT, 0, messageTag, MPI_COMM_WORLD);
        MPI_Ssend(fileName.c_str(), 100, MPI_CHAR, 0, messageTag, MPI_COMM_WORLD);
        MPI_Recv(outFileName, 100, MPI_CHAR, 0, messageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string outFileNameStr = std::string(outFileName);
        cout << outFileNameStr << endl;
        filePath = "image_matrices/results/" + outFileNameStr;
        writeToFile(filePath, nonMaxSuppress, width, height);
    }
}

// We assume that all the processes only have 1GB of memory
int main(int argc, char **argv) {
    
    int rank, world_size;
    int ierr;
    const int overlap = 100;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Root process in charge of coordinating and distributing work
    // Root will also be in charge of side resolution
    if(rank == 0) {
        omp_set_num_threads(3);
        int processing = 1;
        int bigWorkProcessing = 1;

        queue<string> smallWorkQueue;
        queue<string> bigWorkQueue;
        queue<int> processorQueue;
        queue<int> workingQueue;
        queue<int> bigWorkingQueue;
        unordered_map<string, int> bigWorkMap;
        
        // Get the list of smallWorkQueue
        string fileName;
        ifstream smallWorkFile("smallWorkList.txt");

        while(getline(smallWorkFile, fileName)) {
            stringstream ss(fileName);
            string token;
            while(ss >> token) {
                smallWorkQueue.push(token);
            }
        }

        ifstream bigWorkFile("bigWorkList.txt");

        while(getline(bigWorkFile, fileName)) {
            stringstream ss(fileName);
            string token;
            while(ss >> token) {
                bigWorkQueue.push(token);
            }
        }

        for(int i = 1; i < world_size; ++i) {
            processorQueue.push(i);
        }

        int bigWorkSize = bigWorkQueue.size() * 4;
        cout << processorQueue.size() << endl;

        // We should now have a populated list of files to be processed
        // Assumption: the images should be less than 12,000 pixels by 12,000 pixels at its maximum
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // This section is dedicated to work distribution
                string fName = "";
                int *param = new int[5];
                int isTerminate = 0;
                // Do the big work first
                while(bigWorkQueue.size() > 0) {
                    fName = bigWorkQueue.front();

                    for(int i = 0; i < 4; ++i) {
                        while(processorQueue.size() == 0) {
                            // wait
                        }

                        int pRank = processorQueue.front();
                        processorQueue.pop();
                        workingQueue.push(pRank);
                        cout << pRank << endl;
                        bigWorkingQueue.push(pRank);

                        param[0] = 5000;
                        param[1] = 1250;
                        param[2] = i*1250;
                        param[3] = 1;
                        param[4] = 3;
                        MPI_Ssend(&isTerminate, 1, MPI_INT, pRank, 0, MPI_COMM_WORLD);
                        MPI_Ssend(fName.c_str(), 100, MPI_CHAR, pRank, 0, MPI_COMM_WORLD);
                        MPI_Ssend(param, 5, MPI_INT, pRank, 1, MPI_COMM_WORLD);
                    }
                    bigWorkQueue.pop();
                }

                while(smallWorkQueue.size() > 0) {
                    fName = smallWorkQueue.front();
                    smallWorkQueue.pop();

                    // Wait for processors to be available
                    while(processorQueue.size() == 0) {
                        // wait
                    }

                    int pRank = processorQueue.front();
                    processorQueue.pop();
                    workingQueue.push(pRank);

                    param[0] = 512;
                    param[1] = 512;
                    param[2] = 0;
                    param[3] = 0;
                    param[4] = 0;

                    MPI_Ssend(&isTerminate, 1, MPI_INT, pRank, 0, MPI_COMM_WORLD);
                    MPI_Ssend(fName.c_str(), 100, MPI_CHAR, pRank, 0, MPI_COMM_WORLD);
                    MPI_Ssend(param, 5, MPI_INT, pRank, 1, MPI_COMM_WORLD);
                }
            }

            // Need to make the work queue critical sections
            #pragma omp section
            {
                // This thread is just to receive responses and update the processor queue
                while(processing == 1) {
                    if(workingQueue.size() > 0) {
                        // Pop the earliest processor and wait for the reply
                        int pRank = workingQueue.front();
                        workingQueue.pop();

                        int pStatus;
                        MPI_Recv(&pStatus, 1, MPI_INT, pRank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        // Check if there are any pending work that needs to be assigned
                        if(smallWorkQueue.size() == 0 && bigWorkQueue.size() == 0) {
                            // All work has been completed
                            int response = 1;
                            MPI_Ssend(&response, 1, MPI_INT, pRank, 2, MPI_COMM_WORLD);
                            if(workingQueue.size() == 0) {
                                processing = 0;
                            }
                        } else {
                            // More work to be done
                            cout << "There is more work to do" << pRank << endl;
                            int response = 0;
                            processorQueue.push(pRank);
                            MPI_Ssend(&response, 1, MPI_INT, pRank, 2, MPI_COMM_WORLD);
                        }

                        // Return the processor to the processor queue
                    }
                }
            }

            #pragma omp section
            {
                // This section is dedicated to combining the images
                // We can combine more here because we dont need the memory to keep track of multiple matrices
                int pRank;
                cout << "reached" << endl;
                while(bigWorkSize > 0) {
                    // While the queue is empty, wait
                    while(bigWorkingQueue.size() == 0) {
                        //wait
                    }

                    // Check the front of the pRank processor
                    pRank = bigWorkingQueue.front();

                    bigWorkingQueue.pop();
                    bigWorkSize -= 1;

                    int pStatus;
                    char *rankFileName = new char[100];
                    MPI_Recv(&pStatus, 1, MPI_INT, pRank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(rankFileName, 100, MPI_CHAR, pRank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // We get the filename and create a directory with it, we process the images only when the big working queue is
                    // First, strip the file extension ends to get the raw fileName
                    string stringToBeRemoved = ".txt";
                    string rankFileNameStr = std::string(rankFileName);
                    string::size_type iLoc = rankFileNameStr.find(stringToBeRemoved);

                    if(iLoc != string::npos) {
                        rankFileNameStr.erase(iLoc, rankFileNameStr.length());
                    }

                    // Now we should have the raw filename
                    // Update the map with the number of counts and extract the existing count to generate the filename
                    // First, check if we have an entry for the filename
                    unordered_map<string, int>::const_iterator keyIter = bigWorkMap.find(rankFileNameStr);
                    int fileIndex = 0;
                    // If the key doesn;t exist
                    if( keyIter == bigWorkMap.end() ) {
                        bigWorkMap[rankFileNameStr] = 0;
                        fileIndex = 0;
                    } else {
                        fileIndex = keyIter->second + 1;
                        bigWorkMap[rankFileNameStr] = keyIter->second+1;
                    }

                    // Generate the filename
                    rankFileNameStr = rankFileNameStr + "_temp" + to_string(fileIndex) + ".txt";

                    string outFilePath = "/big/" + rankFileNameStr;
                    MPI_Ssend(outFilePath.c_str(), 100, MPI_CHAR, pRank, 3, MPI_COMM_WORLD);
                }

                // Once all the files have been processed, we iterate through the map and start combining the subimages
                // Create a matrix with the big dimensions
                double **imageMat = new2d(5000, 5000);
                size_t size = 5000;
                size *= 5000;
                std::memset(imageMat[0], 0, size);

                for(unordered_map<string, int>::iterator iter = bigWorkMap.begin(); iter != bigWorkMap.end(); ++iter) {
                    string fileNameStr = iter->first;
                    for(int x = 0; x < 4; ++x) {
                        // Read the sub result
                        string filePath = "image_matrices/results/big/" + fileNameStr + "_temp" + to_string(x) + ".txt";
                        ifstream infile(filePath);
                        string line;

                        int i = 0;
                        int j = x*1250;

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

                        if(remove(filePath.c_str()) != 0) {
                            perror("Error deleting file");
                        }
                    }
                    // Once the image matrix is loaded, write it to a final file and delete the temp files
                    string filePath = "image_matrices/results/" + fileNameStr + ".txt";
                    writeToFile(filePath, imageMat, 5000, 5000);
                }

                delete[] imageMat;
            }
        }
        // When we reach here, all the images should be processed and combined
        // If there are any stranded processes in the processor queue free them
        for(int i = 0; i < processorQueue.size(); ++i) {
            int pRank = processorQueue.front();
            processorQueue.pop();
            int isTerminate = 1;
            MPI_Ssend(&isTerminate, 1, MPI_INT, pRank, 0, MPI_COMM_WORLD);
        }
    }else {
        int processing = 1;
        while(processing == 1) {
            // Worker process waiting for work
            // Worker shoudl receive a file name and an array with parameters from the root
            // If no work is left, a message will be sent
            char *fileName = new char[100];
            int *param = new int[5];
            int isTerminate = 0;

            MPI_Recv(&isTerminate, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(isTerminate == 0) {
                MPI_Recv(fileName, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(param, 5, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // Termination called no more jobs available
                processing = 0;
                break;
            }
            // offsetX will represent the x location of the chunk and offset Y will represent the y location of the chunk
            int width = param[0];
            int height = param[1];
            int yOffset = param[2];
            int fileType = param[3]; // 0 for full image, 1 for partial image
            int messageTag = param[4];

            std::string fileNameStr = std::string(fileName);
            processImage(fileNameStr, yOffset, width, height, fileType, messageTag);

            // Send a message to root with tag 2 to request for a new job, check the response to make sure that there is still jobs
            int response = 1;
            MPI_Ssend(&response, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            int rootResponse = 0;
            MPI_Recv(&rootResponse, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(rootResponse == 1) {
                // No more jobs available
                processing = 0;
            }
        }
        cout << "Processor at " << rank << "ended" << endl;
    }
    MPI_Finalize();
}