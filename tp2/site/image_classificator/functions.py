#importacoes gerais
import cv2
import h5py
import numpy as np
import mahotas
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from skimage import io
# from skimage.color import rgb2hsv, rgb2gray
from skimage.exposure import histogram
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern
# from google.colab.patches import cv2_imshow



######importacoes para a classificacao########
import glob
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib


####################################################################################

#funcao teste
def result():
    return "resultado"

####################################################################################

#histograma
# VETOR[512]
def f_histogram(image):
    bins = 8
    # conversão de BGR para HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cálculo do histograma
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalização do histograma
    cv2.normalize(hist, hist)
    feature = hist.flatten()
    return feature 

####################################################################################

#haralick
# VETOR[13]
def f_haralick(image):
    # função que retorna o vetor de características
    haralick = mahotas.features.haralick(image).mean(axis=0)
    return haralick

####################################################################################

#lbp
# VETOR[16]
def f_lbp(image):
    radius = 1
    n_points = 8 * radius
    #   calcula lbp da imagem
    lbp = local_binary_pattern(image, n_points, radius, method = 'uniform')
    #   calcula histograma do lbp
    lbp_hist = cv2.calcHist([image], [0], None, [8], [0, 256])
    lbp_hist = np.array(lbp_hist, dtype=float)
    #   calcula probabilidade
    lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
    feat = np.concatenate((lbp_hist, lbp_prob),axis=0)
    return feat

####################################################################################

#hog
# VETOR[27648]
def f_hog(image): 
    feature, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), 
                            visualize=True, feature_vector = True, multichannel=True)
    #   cv2_imshow(hog_image)
    
    return feature

####################################################################################

#glcm
# VETOR[20]
def f_glcm(image):
    #             greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
    gCoMatrix = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, normed=True)

    #            greycoprops(P, prop='contrast')
    contrast = greycoprops(gCoMatrix, 'contrast')
    dissimilarity = greycoprops(gCoMatrix, 'dissimilarity')
    homogeneity = greycoprops(gCoMatrix, 'homogeneity')
    energy = greycoprops(gCoMatrix, 'energy')
    correlation = greycoprops(gCoMatrix, 'correlation')
    
    feature = np.array([contrast[0][:], dissimilarity[0][:], homogeneity[0][:], energy[0][:], correlation[0][:]])
    feature = feature.flatten()
    return feature

####################################################################################

def define_parameters(feature):
    h5_path = './output/'
    
    #   ajuste das labels
    
    h5f_label = h5py.File(h5_path + 'labels.h5', 'r')
    global_labels_string = h5f_label['dataset_1']
    global_labels = np.array(global_labels_string)

    
    #   DEFINE h5f_data
    if feature == "histogram":
        h5f_data = h5py.File(h5_path + 'data_histogram.h5', 'r')
    
    elif feature == "haralick":
        h5f_data = h5py.File(h5_path + 'data_haralick.h5', 'r')
        
    elif feature == "lbp":
        h5f_data = h5py.File(h5_path + 'data_lbp.h5', 'r')
        
    elif feature == "hog":
        h5f_data = h5py.File(h5_path + 'data_hog.h5', 'r')
        
    elif feature == "glcm":
        h5f_data = h5py.File(h5_path + 'data_glcm.h5', 'r')
    
    
    global_features_string = h5f_data['dataset_1']
    global_features = np.array(global_features_string)

    h5f_data.close()
    h5f_label.close()
    
    test_size = 0.1
    seed = 9
    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                                np.array(global_labels),
                                                                                                test_size=test_size,
                                                                                                random_state=seed)
    return trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal

####################################################################################

def test_image_in_best_model(image_path, feature):
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = define_parameters(feature)
    image_size = tuple((512, 384))
    
    # KNN
    clf = KNeighborsClassifier()
    clf.fit(trainDataGlobal, trainLabelsGlobal)
    
    print(image_path)
    image = cv2.imread('./' + image_path)
    image = cv2.resize(image, image_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if feature == "histogram":
        fv = f_histogram(image)
    
    elif feature == "haralick":
        fv = f_haralick(gray)

    elif feature == "lbp":
        fv = f_lbp(gray)

    elif feature == "hog":
        fv = f_hog(image)

    elif feature == "glcm":
        fv = f_glcm(gray)
    
    global_feature = fv
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    train_labels = ["agata potato", "asterix potato", "cashew", "diamond peach", "fuji apple", "granny smith apple", "honney drew melon", "kiwi", "nectarine", "onion", "orange", "plum", "spanish pear", "taiti lime", "watermelon"]

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # # display the output image
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()

    return train_labels[prediction]

####################################################################################