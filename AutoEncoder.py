import theanets as the
import climate
import matplotlib.pyplot as plt
import numpy as np
import cPickle

##TODO: make the size automatic
#climate.enable_default_logging()

__model = 0


def encoding(my_data):

    my_data_encoded = __model.encode(my_data)


    return my_data_encoded


def decoding(data_encoded):
    return __model.decode(data_encoded)


def load_training():
    global __model
    __model = the.Autoencoder([768, (100, 'sigmoid'), (768, 'tied', 'sigmoid')])

    return True


def get_weights():

    with open('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/30000id/l1_norm/weights_hd01_30000samples_hiddenl105.txt', 'rb') as handle:
        weights = cPickle.load(handle)

    return weights


def load_deep_AE():
    global __model

    __model = the.Autoencoder(layers=(768,400,(100,'sigmoid'),('tied',400,'sigmoid'),('tied',768,'sigmoid')))

    return True


def get_stacked_weigths(structure):

    #with open('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/CHOT_pedestrian_stacked_sparsity002.txt', 'rb') as handle:
    with open('C:/Users/dario.dotti/Documents/tracking_points/tracking_data_kinect2/matrix_precomputed/exp_june/CHOT_indoor_stacked_unit_sparsity002.txt','rb') as handle:
        list_weights = cPickle.load(handle)

    return list_weights