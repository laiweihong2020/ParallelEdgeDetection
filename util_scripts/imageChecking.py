from PIL import Image
import numpy as np

def main():
    # im = Image.open('../images/lena512color.tif').convert('L')
    # # Convert the grayscale image into an numpy array to be processed in C++
    # I = np.asarray(im, dtype=int)
    # np.savetxt('../image_matrices/lena512.txt', I, fmt='%d')
    I = np.loadtxt('../image_matrices/results/lena.txt')
    im = Image.fromarray(np.uint8(I))
    d = im.resize((512, 512))
    d.show()

if __name__ == '__main__':
    main()