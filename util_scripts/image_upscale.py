from PIL import Image
import numpy as np

def main():
    im = Image.open('../images/lena512color.tif').convert('L')
    d = im.resize((5000, 5000), resample=Image.BOX)
    I = np.asarray(d, dtype=int)
    np.savetxt('../image_matrices/lena.txt', I, fmt='%d')

if __name__ == '__main__':
    main()