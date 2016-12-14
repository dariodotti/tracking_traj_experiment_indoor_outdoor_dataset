import pickle
import numpy as np
import cv2

import experiments as my_exp
import AutoEncoder as my_ae








def main():


    with open('C:/Users/dario.dotti/Documents/tracking_points/tracking_data_kinect2/matrix_precomputed/exp_june/cahot_tr_23_5_20_4_20_5.txt', 'rU') as handle:
        SPEED_training_matrix = pickle.load(handle)

    with open('C:/Users/dario.dotti/Documents/tracking_points/tracking_data_kinect2/matrix_precomputed/exp_june/cahot_te_23_5_20_4_20_5.txt', 'rU') as handle:
        SPEED_test_matrix = pickle.load(handle)

    all_data_matrix = np.vstack((SPEED_training_matrix,SPEED_test_matrix))

    with open('C:/Users/dario.dotti/Documents/tracking_points/tracking_data_kinect2/matrix_precomputed/exp_june/chot_23_encoded_data.gz', 'rb') as handle:
       AE_matrix = pickle.load(handle)
    AE_training = AE_matrix[:len(SPEED_training_matrix)]
    AE_test=AE_matrix[len(SPEED_training_matrix):]

    cluster_prediction = my_exp.cluster_fit_predict_kmeans(all_data_matrix,k=3)

    training_labels= cluster_prediction[:len(SPEED_training_matrix)]
    test_labels= cluster_prediction[:len(SPEED_test_matrix)]

    #train Logistic regression classifier
    my_exp.logistic_regression_train(AE_training,training_labels)

    #test
    pred = my_exp.logistic_regression_predict(AE_test)

    #print pred

    counter = 0

    for i in range(0,len(pred)):
        if pred[i]==test_labels[i]:
            counter+=1

    print float(counter)/len(test_labels)













if __name__ == '__main__':
    main()