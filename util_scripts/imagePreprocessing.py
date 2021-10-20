from PIL import Image
import numpy as np 
from os import listdir
from os.path import isfile, join

def main():
    # Get the directory and process the images accordingly
    files = [f for f in listdir('../images/') if isfile(join('../images/', f))]
    bF = open("../bigWorkList.txt", "w")
    sF = open("../smallWorkList.txt", "w")

    # Process them
    number_of_big_files = 5

    for f in files:
        if number_of_big_files > 0:
            number_of_big_files = number_of_big_files - 1
            imagePath = "../images/" + f
            im = Image.open(imagePath).convert('L')
            d = im.resize((5000, 5000), resample=Image.BOX)
            I = np.asarray(d, dtype=int)
            f = f.replace('.jpg', '').replace('.tif', '')
            outPath = "../image_matrices/" + f + ".txt"
            np.savetxt(outPath, I, fmt='%d')
            
            # Attach the image to the big text list
            bF.write(f)
            bF.write("\n")

        else:
            imagePath = "../images/" + f
            im = Image.open(imagePath).convert('L')
            d = im.resize((512, 512), resample=Image.BOX)
            I = np.asarray(d, dtype=int)
            f = f.replace('.jpg', '').replace('.tif', '')
            outPath = "../image_matrices/" + f + ".txt"
            np.savetxt(outPath, I, fmt='%d')

            sF.write(f)
            sF.write("\n")

    bF.close()
    sF.close()

if __name__ == '__main__':
    main()