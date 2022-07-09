from ELA_Preprocess import ELA_preprocessing
import pickle
import numpy as np
import os

class cl_inference():
   def pan_inference(self, imagepath, model_path):
        '''

        :param imagepath: Path of the image directory
        :param model_path: Path of the trained model
        :return: Result stating if the PAN card is fake or real
        '''
        # load the model from disk
        Preprocess_obj = ELA_preprocessing()
        elaimg = np.array(Preprocess_obj.convert_to_ela_image(imagepath, 95)).reshape(-1,1)
        #print(elaimg)
        loaded_model = pickle.load(open(model_path, 'rb'))
        result = loaded_model.predict(elaimg)
        if result == 0:
             return "Fake"
        else :
             return "Real"



pan_obj = cl_inference()
#res = pan_obj.pan_inference(r"C:\Users\dilna\Downloads\MicrosoftTeams-image (7).png",r"C:\Users\dilna\PycharmProjects\PanCardProject\best_vgg19_10_06_2022-12_50_28.h5")
#print("The PAN Card is",res)
folder = r"C:\Users\dilna\OneDrive\Desktop\Testing_PANcardPOC\Real"
for filename in os.listdir(folder):
     img_path = os.path.join(folder,filename)
     res = pan_obj.pan_inference((img_path),r"C:\Users\dilna\PycharmProjects\PanCardProject\pan_card_log_reg_model2.sav")
     print(filename, "- ",res)
