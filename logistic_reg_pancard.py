import os
from ELA_Preprocess import ELA_preprocessing
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay



#Training a PAN Card Fraud Detection Model
class PANcardmodel():
    #Spliting into training and testing datasets

    def datasetsplit(self,img_dir):
        '''

        :param img_dir: the path of a folder thats contains folders of fake and real PAN card images
        :return: Training and Testing datasets with their labels
        '''

        Preprocess_obj = ELA_preprocessing() #Preprocessing including conversion of images into its ELA
        CATEGORIES = os.listdir(img_dir) #CATEGORIES - Real and Fake
        X = [] #Max_Diff
        Y = []

        for category in CATEGORIES:
            path = os.path.join(img_dir, category)
            classIndex = CATEGORIES.index(category) # 0- Fake 1-Real

            for imgs in os.listdir(path):
                imgpath = os.path.join(path, imgs)
                X.append(Preprocess_obj.convert_to_ela_image(imgpath, 95))
                Y.append(classIndex)

        X = np.array(X).reshape(-1,1)
        Y = np.array(Y)

        #Splitting the data into training and testing datasets
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=5, shuffle=True)

        #Feature Scaling
        #sc_X = StandardScaler()
        #X_train = sc_X.fit_transform(X_train)
        #X_val = sc_X.transform(X_val)

        return X_train, X_val, Y_train, Y_val

    #Fitting a model and returning the trained model
    def model_train(self, img_dir):
        '''
        :param img_dir: the path of a folder thats contains folders of fake and real PAN card images
        :return: Trained PAN card model
        '''

        X_train, X_val, Y_train, Y_val = self.datasetsplit(img_dir)

        classifier = LogisticRegression(C=1, penalty='l2', solver='liblinear' ,random_state=0)
        classifier.fit(X_train, Y_train)

        Y_Pred = classifier.predict(X_val)
        cm = confusion_matrix(Y_val, Y_Pred)
        print(metrics.classification_report(Y_val, Y_Pred, target_names=["Fake", "Real"]))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = classifier.classes_)
        disp.plot()
        plt.show()

        #saving the model
        filename = 'pan_card_log_reg_model3.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        print("Model Saved")

        return classifier

clsobj = PANcardmodel()
model = clsobj.model_train( r"C:\Users\dilna\OneDrive\Desktop\PanCardPOC\Augmented for ELA")







