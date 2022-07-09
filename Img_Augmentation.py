# Pan card fraud detection- DataFlair
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


img_size = 500
datadir = r'C:\Users\dilna\OneDrive\Desktop\images1cropped' # root data directory
CATEGORIES = os.listdir(datadir)
print(CATEGORIES)


#Data Augmentation starts here
def PreProcess(img_size, path):
    """This function reads images from the given folders subfolder
      and returns a normalized array along with their respective classes"""
    x, y = [], []
    CATEGORIES = os.listdir(path)
    print("Found {} classes: {}".format(len(CATEGORIES), CATEGORIES))
    c=1

    for category in CATEGORIES:
        path = os.path.join(datadir, category)
#        classIndex = CATEGORIES.index(category)

        for imgs in tqdm(os.listdir(path)): #Tqdm is a loop progress visualizer
            img_arr = cv2.imread(os.path.join(path, imgs))
            filename = str(r"C:\Users\dilna\OneDrive\Desktop\PanCardPOC\Augmented Images\\")+str(c)+str(".png")

            # resize the image
            resized_array = cv2.resize(img_arr, (img_size, img_size))
            cv2.imshow("images", resized_array)
            cv2.waitKey(1)
            cv2.imwrite(filename, resized_array)
            c += 1
#            y.append(classIndex)
            # Rotation around 180 degree
            rotation = cv2.rotate(resized_array, cv2.ROTATE_180)
            cv2.imshow("Rotated", rotation)
            cv2.waitKey(1)
            filename = str(r"C:\Users\dilna\OneDrive\Desktop\PanCardPOC\Augmented Images\\") + str(c) + str(".png")
            cv2.imwrite(filename, rotation)
            c += 1
            rotation = rotation / 255.0
            x.append(rotation)
#            y.append(classIndex)
            # Rotation around 90 degree clockwise
            rotation = cv2.rotate(resized_array, cv2.cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("Rotated", rotation)
            cv2.waitKey(1)
            filename = str(r"C:\Users\dilna\OneDrive\Desktop\PanCardPOC\Augmented Images\\") + str(c) + str(".png")
            cv2.imwrite(filename, rotation)
            c += 1
            rotationclock = rotation / 255.0
            x.append(rotationclock)
#            y.append(classIndex)
            # Rotation around 90 degree counter-clockwise
            rotation = cv2.rotate(resized_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow("Rotated", rotation)
            cv2.waitKey(1)
            filename = str(r"C:\Users\dilna\OneDrive\Desktop\PanCardPOC\Augmented Images\\") + str(c) + str(".png")
            cv2.imwrite(filename, rotation)
            c += 1
            rotationcntclock = rotation / 255.0
            x.append(rotationcntclock)
#            y.append(classIndex)

            # Horizontal Flipping
            horizontal_flip = cv2.flip(resized_array, 1)
            cv2.imshow(r"hflip", horizontal_flip)
            cv2.waitKey(1)
            filename = str(r"C:\Users\dilna\OneDrive\Desktop\PanCardPOC\Augmented Images\\") + str(c) + str(".png")
            cv2.imwrite(filename, horizontal_flip)
            c += 1
            horizontal_flip = horizontal_flip / 255.0
            x.append(horizontal_flip)
#            y.append(classIndex)
            # Vertical Flipping
            vertical_flip = cv2.flip(resized_array, 0)
            cv2.imshow(r"vflip", vertical_flip)
            cv2.waitKey(1)
            filename = str(r"C:\Users\dilna\OneDrive\Desktop\PanCardPOC\Augmented Images\\") + str(c) + str(".png")
            cv2.imwrite(filename, vertical_flip)
            c += 1
            vertical_flip = vertical_flip / 255.0
            x.append(vertical_flip)
#            y.append(classIndex)
            # Vertical and Horizontal Flipping
            vh_flip = cv2.flip(resized_array, -1)
            cv2.imshow(r"vhflip", vh_flip)
            cv2.waitKey(1)
            filename = str(r"C:\Users\dilna\OneDrive\Desktop\PanCardPOC\Augmented Images\\") + str(c) + str(".png")
            cv2.imwrite(filename, vh_flip)
            c += 1
            vh_flip = vh_flip / 255.0
            x.append(vh_flip)
        #y.append(classIndex)


    cv2.destroyAllWindows()
    return x

x= PreProcess(img_size, datadir)

print(len(x))
