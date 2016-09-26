import argparse
import cv2
import numpy as np
import pickle
from collections import Counter
import random
from sklearn.metrics import normalized_mutual_info_score as NMI

import img_proc as my_img_proc
import main as main_camera017
import experiments as my_exp
import AutoEncoder as my_AE




def main():

    ##divide image into patches(polygons) and get the positions of each one
    scene = cv2.imread('C:/Users/dario.dotti/Documents/LOST_dataset/camera001.jpg')
    list_poly = my_img_proc.divide_image(scene)



    with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/raw_traj_organized_by_id.txt', 'rb') as handle:
        slices = pickle.load(handle)
    print 'data extracted'

    ##Feature Extraction METHOD: compute histogram of Oriented Tracklets for all the tracklets in every patch
    normalized_data_HOT,orig_list_id = main_camera017.histograms_of_oriented_trajectories(list_poly,slices)

    ##Feature Extraction METHOD: Compute the features velocity-acceleration-curvature to distinguish between pedestrian and cars
    #normalized_data_HOT = velo_acc_curva_feature(list_poly,slices)

    ####Experinment AUTOENCODER
    # ##3)load the model of the trained AE
    # if my_AE.load_training():
    #     print 'model loaded'
    #     weights = my_AE.get_weights()
    #     weights= weights.T

    ###)3a)load the model of the trained deep AE
    if my_AE.load_deep_AE():
        print 'model loaded'
        list_weights = my_AE.get_stacked_weigths([1,2])

        hid1_space = np.dot(normalized_data_HOT,list_weights[0])
        #hid2_space = np.dot(hid1_space,list_weights[1])


        weights= list_weights[1].T



    ####COMPUTE THE MOST SIMILAR FEATURE FOR EVERY WEIGHT

    #mostSimilar_feature_sorted,seeds_index = my_exp.get_most_similar_feature_from_AEweights(hid1_space,weights,orig_list_id,scene,slices)


    ###CLUSTERING

    ####seed
    # seeds = []
    # # ##get the traj with strongest weight and use it to initialize mean shift
    # map(lambda s:seeds.append(normalized_data_HOT[s]),seeds_index)
    # #
    # cluster_pred_AE,cluster_centers_seeds = my_exp.cluster_fit_predict(normalized_data_HOT,seeds)
    # freq_counter_seedsAE =  Counter(cluster_pred_AE)
    #
    #
    # print 'seeds from AE'
    # print freq_counter_seedsAE
    # #
    # main_camera017.color_traj_based_on_clusters(slices,cluster_pred_AE,freq_counter_seedsAE,scene)
#############



     ##raw features cluster
    cluster_pred_raw,cluster_centers_unseeds = my_exp.cluster_fit_predict(normalized_data_HOT,[])
    freq_counter_unseed =  Counter(cluster_pred_raw)
    print freq_counter_unseed

    # ##getting same number of cluster with raw features
    # temp_seeds = []
    # for trj_id in freq_counter_unseed.keys()[:len(freq_counter_seedsAE.keys())] :
    #     ##get index of every class
    #     index = np.where(cluster_pred_raw == trj_id)[0][0]
    #
    #     temp_seeds.append(normalized_data_HOT[index])
    #
    # cluster_pred_raw,cluster_centers_unseeds = my_exp.cluster_fit_predict(normalized_data_HOT,temp_seeds)
    # freq_counter_seeds_raw =  Counter(cluster_pred_raw)
    # print 'seeds from raw features'
    # print freq_counter_seeds_raw

    main_camera017.color_traj_based_on_clusters(slices,cluster_pred_raw,freq_counter_unseed,scene)

    ####SUPERVISED CLASSIFIER
    test_samples = []
    test_labels= []

    training_samples = []
    training_labels= []
    test_pos = []


    ##get test samples from every class
    for trj_c,trj_id in enumerate(freq_counter_unseed):
        ##get index of every class

        index = np.where(cluster_pred_raw == trj_id)[0]
        #take 10% of every class as test samples
        #print w[trj_c]
        random_index = random.sample(index,int((len(index)*0.15)))

        for i,c in enumerate(cluster_pred_raw):
            if i in random_index:
                test_samples.append(normalized_data_HOT[i])
                test_labels.append(trj_id)
                test_pos.append(i)
    print len(test_samples)

    for i,sample in enumerate(normalized_data_HOT):
        if i not in test_pos:
            training_samples.append(sample)
            training_labels.append(cluster_pred_raw[i])
    print np.array(training_samples).shape


    # #train Logistic regression classifier
    my_exp.logistic_regression_train(training_samples,training_labels)

    # #test
    pred = my_exp.logistic_regression_predict(test_samples)

    print NMI(test_labels,pred)
    #
    # counter = 0
    #
    # for i in range(0,len(pred)):
    #     if pred[i]==test_labels[i]:
    #         counter+=1
    #
    # print float(counter)/len(test_labels)


if __name__ == '__main__':
    main()