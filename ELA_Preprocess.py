import sys
from PIL import Image, ImageChops, ImageEnhance

#Preprocessing of PAN Card Images
class ELA_preprocessing():

    #Function to convert the image into its ELA
    def convert_to_ela_image(self, path, quality):
        '''

        :param path: the path of a folder thats contains folders of fake and real PAN card images
        :param quality: by default 95
        :return: ELA images
        '''
        filename = sys.argv[0]
        resaved = filename + '.resaved.jpg'
        #ela = filename + '.ela.png'
        # Converting all the images to RGB format
        im = Image.open(path).convert('RGB')

        # Resaving the images to JPEG format
        im.save(resaved, 'JPEG', quality=quality)
        resaved_im = Image.open(resaved)

        # Finding the difference between the original image and its ELA
        ela_im = ImageChops.difference(im, resaved_im)
        # Gets the minimum and maximum pixel values for each band in the image.
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
        return max_diff
