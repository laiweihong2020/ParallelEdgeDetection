from PIL import Image
import numpy as np

def main():
    im = Image.open('../images/lena512color.tif').convert('L')
    im.show();
    # Convert the grayscale image into an numpy array to be processed in C++
    I = np.asarray(im, dtype=int)
    np.savetxt('../image_matrices/lena512.txt', I, fmt='%d')

if __name__ == '__main__':
    main()