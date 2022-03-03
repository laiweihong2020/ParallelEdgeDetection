# Parallel Edge Detection
In this project, I present a solution that attempts to parallelise the operations of Canny Edge Detectors by optimising job scheduling. Refer to the submission folder for the final implementation.

## Compiled with
C++11

## Instructions to run
Precondition: All images have to be converted into image matrices that with corresponding pixel value at each location. Refer to imagePreprocessing.py for details on operations.

Running: 
1) Insert file names based on the nature of the file. In this case, 512x512 for small work lists and 5000x5000 for big work lists. 
2) Simply compile and run the program. The directories to read and output are hardcoded in this instance.
```
g++ par_canny.cpp -o par_canny.out
./par_canny.out
```
3) The output should be at image_matrices/results/ with the same file name

## Important note
Many aspects of this project is hard coded and no plans are made to extend this project.