import argparse
import cv2
import numpy as np
import copy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import cPickle as cpickle
import pickle
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
import sklearn.metrics.pairwise as similarity
import random
import scipy.io as IO


import img_proc as my_img_proc
import experiments as my_exp
import AutoEncoder as my_AE



#Global variable shared by all functions
file_content = []
ids = []

def read_data_fromEachFile_timeIntervals(file):

    with open(file,'r')as f:
            file_content = f.read().split('\n')

    fps = 5.95
    time_interval = 300

    #### convert frame in seconds###
    content_time = map(lambda line: int(float(line.split(' ')[1])/fps),file_content)
    #print content_time

    time_slices = []
    time_slices_append = time_slices.append


    #######Split file according to periods of time#####
    for t in xrange(time_interval,content_time[len(content_time)-1]+1,time_interval):
        list_time_interval = []
        list_time_interval_append = list_time_interval.append

        for i in xrange(0,len(content_time)):
            if content_time[i] <= t and content_time[i] > t-time_interval:
                list_time_interval_append(file_content[i])

        time_slices_append(list_time_interval)


    return time_slices,file_content


def load_matrix_pickle(filename):

    with open(filename, 'rb') as handle:
        file = cpickle.load(handle)
    return file


def save_matrix_pickle(file,filename):

    with open(filename, 'wb') as handle:
        cpickle.dump(file,handle,protocol=2)


def my_function(k):
    print k
    temp_track = []
    temp_track_append= temp_track.append
    map(lambda i: temp_track_append(file_content[i]) if ids[i] == k else False,xrange(len(file_content)))

    return temp_track


def read_data_tracklets(file,multiThread):

    global file_content
    with open(file,'r')as f:
            file_content = f.read().split('\n')

    global ids
    ids =map(lambda line: int(line.split(' ')[0]),file_content)
    keys = list(set(ids))
    #data is sorted only if multithread isnt used
    keys = sorted(keys,key=lambda x: x)
    print len(keys)
    #keys= keys[:10000]

    ####MULTI-THREAD VERSION######
    if multiThread:
        cores = 6
        pool = ThreadPool(cores)
        print 'n cores: '+str(cores)

        tracklets = pool.map(lambda k: my_function(k) ,keys)

        #close the pool and wait for the work to finish
        pool.close()
        pool.join()
    ###########################
    else:
        # keys= keys[:500]
        tracklets = []
        tracklets_append = tracklets.append

        for k in keys:
            #print k
            temp_track = []
            temp_track_append= temp_track.append

            map(lambda i: temp_track_append(file_content[i]) if ids[i] == k else False,xrange(len(file_content)))

            tracklets_append(temp_track)


    return tracklets


def get_coordinate_points(occurance):

    xs = map(lambda line: int(float(line.split(' ')[2])),occurance)
    ys = map(lambda line: int(float(line.split(' ')[3])),occurance)
    ids =map(lambda line: str(line.split(' ')[0]),occurance)


    #list_points = []
    #list_points_append = list_points.append
    #map(lambda c: list_points_append((xs[c],ys[c])),xrange(0,len(xs)))
    #apply filter to cancel noise
    #x_f,y_f =my_img_proc.median_filter(list_points)

    return xs,ys,ids


def trajectories_in_interval(my_room, list_poly, slices):

    print 'Region Based Hist'

    #get number of patches
    slice_col = my_img_proc.get_slice_cols()
    slice_row = my_img_proc.get_slice_rows()

    #counter for every id should be empty
    track_points_counter = np.zeros((slice_col*slice_row))

    my_data_temp = []
    my_data_temp_append = my_data_temp.append


    for i in xrange(0, len(slices)):

        #get x,y,z of every traj point after smoothing process
        x_f,y_f,ids = get_coordinate_points(slices[i])


        #count the occurances of filtered point x,y in every patches
        for p in xrange(0,len(list_poly)):
            for ci in xrange(0,len(x_f)):
                #2d polygon
                if list_poly[p].contains_point((int(x_f[ci]),int(y_f[ci]))):
                        track_points_counter[p] = track_points_counter[p]+1

        #count the occurances of filtered point x,y in every patches: more efficent way
        # map(lambda p: map(lambda i:track_points_counter.__setitem__(p,track_points_counter[p]+1)  \
        #     if list_poly[p].contains_point((int(x_f[i]),int(y_f[i]))) else False ,xrange(len(list_poly))), xrange(len(list_poly)))

        #save the data of every group in the final matrix
        my_data_temp_append(track_points_counter)

        show = False
        if show:
            temp_img = copy.copy(my_room)

            # #draw filtered points
            for p in range(0,len(x_f)):
                    cv2.circle(temp_img,(int(x_f[p]),int(y_f[p])),2,255,-1)#my_room

            ##draw slices
            for p in range(0,len(list_poly)):
               cv2.rectangle(temp_img,(int(list_poly[p].vertices[1][0]),int(list_poly[p].vertices[1][1])),\
                           (int(list_poly[p].vertices[3][0]),int(list_poly[p].vertices[3][1])),0,1)

            cv2.imshow('lab_room',temp_img)
            cv2.waitKey(0)

    ## normalize the final matrix
    normalized_finalMatrix = np.array(normalize(np.array(my_data_temp),norm='l2'))
    print 'final matrix size'
    print normalized_finalMatrix.shape
    ### visualizing final histograms of tracklets
    #visualize_weights(list_poly,normalized_finalMatrix)


    return normalized_finalMatrix


def histograms_of_oriented_trajectories(list_poly,slices):
    hot_all_data_matrix = []
    hot_all_data_matrix_append = hot_all_data_matrix.append

    print 'HOT FEATURES'

    temp_list_id = []
    for i in xrange(0, len(slices)):
        #get x,y,z of every traj point after smoothing process
        x_f,y_f,ids = get_coordinate_points(slices[i])
        temp_list_id.append(ids[0])

        ##initialize histogram of oriented tracklets
        hot_matrix = []

        ###store all the tracklets for every patch
        for p in xrange(0,len(list_poly)):
            tracklets_in_cube = []
            tracklet_in_cube_append = tracklets_in_cube.append


            ##Array for HOT
            map(lambda ci:tracklet_in_cube_append([x_f[ci],y_f[ci],ids[ci]]) if list_poly[p].contains_point((int(x_f[ci]),int(y_f[ci])))\
                else False ,xrange(len(x_f)))


            for tracklet in [tracklets_in_cube]:
                if len(tracklet)>0:

                    #for tracklet in patch compute HOT following paper
                    hot_single_poly = my_img_proc.histogram_oriented_tracklets(tracklet)

                    #compute hot+curvature
                    #hot_single_poly = my_img_proc.histogram_oriented_tracklets_plus_curvature(tracklet)

                else:
                    hot_single_poly = np.zeros((24))

                ##add to general matrix
                if len(hot_matrix)>0:
                    hot_matrix = np.hstack((hot_matrix,hot_single_poly))
                else:
                    hot_matrix = hot_single_poly
        hot_all_data_matrix_append(hot_matrix)


    ## normalize the final matrix
    normalized_finalMatrix = np.array(normalize(np.array(hot_all_data_matrix),norm='l2'))

    print 'final matrix size'
    print normalized_finalMatrix.shape

    ### visualizing final histograms of tracklets
    #visualize_weights(list_poly,normalized_finalMatrix)

    return normalized_finalMatrix,temp_list_id


def velo_acc_curva_feature(list_poly,slices):
    speed_all_data_matrix = []
    speed_all_data_matrix_append = speed_all_data_matrix.append

    print 'SPEED FEATURES'

    temp_list_id = []
    for i in xrange(0, len(slices)):
        #get x,y,z of every traj point after smoothing process
        x_f,y_f,ids = get_coordinate_points(slices[i])
        temp_list_id.append(ids[0])

        ##initialize histogram of oriented tracklets
        speed_matrix = []

        ###store all the tracklets for every patch
        for p in xrange(0,len(list_poly)):
            tracklets_in_cube = []
            tracklet_in_cube_append = tracklets_in_cube.append

            ##append info in the patches they belong
            map(lambda ci:tracklet_in_cube_append([x_f[ci],y_f[ci],ids[ci]]) if list_poly[p].contains_point((int(x_f[ci]),int(y_f[ci])))\
                else False ,xrange(len(x_f)))

            for tracklet in [tracklets_in_cube]:
                if len(tracklet)>0:

                    if p == 2:
                        need_correction=True
                    else:
                        need_correction=False

                    #for tracklet in Cube compute the new feature speed(velocity,Acceleration,curvature)
                    speed_single_poly = my_img_proc.get_velocity_curvature_acceleration(tracklet,need_correction)

                else:

                    speed_single_poly = np.zeros((3))

                ##add to general matrix
                if len(speed_matrix)>0:
                    speed_matrix = np.hstack((speed_matrix,speed_single_poly))
                else:
                    speed_matrix = speed_single_poly

        speed_all_data_matrix_append(speed_matrix)

    ## normalize the final matrix
    normalized_finalMatrix = np.array(normalize(np.array(speed_all_data_matrix),norm='l2'))

    print 'final matrix size'
    print normalized_finalMatrix.shape
    return normalized_finalMatrix


def visualize_weights(list_poly,normalized_finalMatrix):
    tot_frequencty = np.zeros([1,normalized_finalMatrix.shape[1]])

    for i in range(0,normalized_finalMatrix.shape[0]):
        tot_frequencty = tot_frequencty+ normalized_finalMatrix[i]

    average_frequency = tot_frequencty/normalized_finalMatrix.shape[0]

    ##for HOT matrix
    #average_frequency = np.sum(np.reshape(average_frequency,(32,24)),axis=1)

    #Update image every row
    my_freq_hist = np.ones((480,640),dtype=np.uint8)
    my_freq_hist.fill(255)
    ax = plt.subplot()
    plt.title('far distance activation')

    for p in range(0,len(list_poly)):
        # ##draw slices for close distance
        if average_frequency[0][p] <= 0.0:
            continue
        cv2.rectangle(my_freq_hist,(int(list_poly[p].vertices[1][0]),int(list_poly[p].vertices[1][1])),\
                      (int(list_poly[p].vertices[3][0]),int(list_poly[p].vertices[3][1])),(int((1-average_frequency[0][p])*255),0,0),-1)#(1-far_dist[p])


    # ##Show image with points per group
    plt.imshow(my_freq_hist,cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
    plt.show()


def color_traj_based_on_clusters(slices,cluster_pred,freq_counter,scene):

    ###visualize cluster
    # for i,x in enumerate(freq_counter.most_common()):
    #     temp_cluster = []
    #     for j,item in enumerate(cluster_pred):
    #         if x[0]==item:
    #             temp_cluster.append(slices[j])
    #
    #     print 'label '+str(x[0]),'frequency '+str(x[1])
    #     visualize_weights(list_poly,np.array(temp_cluster))

    classes= freq_counter.keys()[:30]

    cm_subsection = np.linspace(0.0, 1.0, len(classes))
    colors = [ plt.cm.jet(x) for x in cm_subsection ]

    #classes = np.sort(classes)
    for trj_id in classes:
        implot = plt.imshow(scene)
        ##get index of every class
        index = np.where(cluster_pred == trj_id)[0]

        print freq_counter.most_common()[trj_id]
        for i in index[:50]:

            x_f,y_f,ids = get_coordinate_points(slices[i])

            for p in xrange(0,len(x_f)):
                plt.scatter(int(x_f[p]),int(y_f[p]),color=colors[trj_id],s=5)


        plt.show()


def show_normal_abnormal_traj(slices,scene):
    x_f,y_f,ids = get_coordinate_points(slices)
    implot = plt.imshow(scene)

    tmp_img=scene.copy()
    for p in xrange(0,len(x_f)):

        cv2.circle(tmp_img,(int(x_f[p]),int(y_f[p])),2,(255,0,),-1)
        #plt.scatter(int(x_f[p]),int(y_f[p]),color='red',s=5)


    cv2.imshow('camera 017',tmp_img)
    cv2.waitKey(0)

def normal_abnormal_class_division(freq_counter,cluster_pred,data,abnormal_class):
    normal_pedestrian_class=[]
    abnormal_pedestrian_class=[]

    for traj in freq_counter.keys()[:30]:
        if traj in abnormal_class:
            index = np.where(cluster_pred == traj)[0]
            for i in index:
                #print i
                abnormal_pedestrian_class.append(data[i])

        else:
            index = np.where(cluster_pred == traj)[0]
            for i in index:
                normal_pedestrian_class.append(data[i])
                # show_normal_abnormal_traj(pedestrian_cluster[i],scene)
    return normal_pedestrian_class,abnormal_pedestrian_class

def main():

    # ##INPUT: path of the txt file with all the recorded days
    # parser = argparse.ArgumentParser(description='path to txt file')
    # parser.add_argument('path_toData')
    #
    # args=parser.parse_args()
    # path_todata = args.path_toData

    ##divide image into patches(polygons) and get the positions of each one
    scene = cv2.imread('C:/Users/dario.dotti/Documents/LOST_dataset/camera017.jpg')
    list_poly = my_img_proc.divide_image(scene)



    ##----------------------------------------------------------------------------------------------------------------##
    ##1a)divide the data in slices using time
    #slices,content = read_data_fromEachFile_timeIntervals(path_todata)

    ##1b)divide the data in slices using id
    #slices = read_data_tracklets(path_todata,multiThread=1)
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/single_files/raw_traj_organized_by_id.txt', 'wb') as handle:
    #     pickle.dump(slices,handle)
    #
    # return False
    #
    #
    # ##1c) load the tracklets already extracted from a pickle file
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/raw_traj_organized_by_id.txt', 'rb') as handle:
    #     slices = pickle.load(handle)
    # print 'raw traj ID extracted'
    #
    # slices = slices[:100]


    ##----------------------------------------------------------------------------------------------------------------##
    ##2a)Feature Extraction METHOD: TAKE points and create region based hist
    #normalized_RBH= trajectories_in_interval(scene, list_poly, slices)

    ##2b)Feature Extraction METHOD: compute histogram of Oriented Tracklets for all the tracklets in every patch
    #normalized_data_HOT,orig_list_id = histograms_of_oriented_trajectories(list_poly,slices)

    # ##2ba)SAVE HOT data
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/30000id/30000_data_HOT.gz', 'wb') as handle:
    #     cpickle.dump(normalized_data_HOT,handle,protocol=2)

    # ##2bb)Load HOT data
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/30000id/30000_data_HOT.gz', 'rb') as handle:
    #     normalized_data_HOT = cpickle.load(handle)
    #     #normalized_data_HOT = normalized_data_HOT[:20000]
    #     orig_list_id = []
    #     map(lambda line: orig_list_id.append(line[0].split(' ')[0]),slices)
    # print 'HOT faetures extracted'

    ##2c)Feature Extraction METHOD: Compute the features velocity-acceleration-curvature to distinguish between pedestrian and cars
    #normalized_data_SPEED = velo_acc_curva_feature(list_poly,slices)



##--------------------------------------------------------------------------------------------------------------------##
####Experinment PEDESTRIAN VS CARS ABNORMAL DETECTION
    #pedestrian_cluster,cars_cluster = my_exp.cars_vs_pedestrian(normalized_data_SPEED,slices,scene)

    pedestrian_cluster=load_matrix_pickle('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/pedestrian_cluster.txt')

    orig_list_id = []
    map(lambda line: orig_list_id.append(line[0].split(' ')[0]),pedestrian_cluster)


    # cars_cluster=load_matrix_pickle('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/car_cluster.txt')
    #
    # orig_list_id = []
    # map(lambda line: orig_list_id.append(line[0].split(' ')[0]),cars_cluster)


    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/car_cluster.txt', 'wb') as handle:
    #     pickle.dump(cars_cluster,handle)
    #
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cluster.txt', 'wb') as handle:
    #     pickle.dump(pedestrian_cluster,handle)


##----------------------------------------------------------------------------------------------------------------##


######################

    ##get pedestrian data

    #normalized_data_HOT_pedestrian,orig_list_id = histograms_of_oriented_trajectories(list_poly,pedestrian_cluster)
    #normalized_data_HOT_cars,orig_list_id = histograms_of_oriented_trajectories(list_poly,cars_cluster)

    #save_matrix_pickle(normalized_data_HOT_pedestrian,'C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/HOT_pedestrian.gz')
    #save_matrix_pickle(normalized_data_HOT_cars,'C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/HOT_cars.gz')

    #normalized_data_SPEED_pedestrian = velo_acc_curva_feature(list_poly,pedestrian_cluster)

    normalized_data_HOT_pedestrian=load_matrix_pickle('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/CHOT_pedestrian.gz')

    #normalized_data_HOT_cars = load_matrix_pickle('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/CHOT_cars.gz')

    #####
##----------------------------------------------------------------------------------------------------------------##

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

        hid1_space = np.dot(normalized_data_HOT_pedestrian,list_weights[0])
        #hid2_space = np.dot(hid1_space,list_weights[1])


        weights= list_weights[1].T
        print weights.shape



    ####COMPUTE THE MOST SIMILAR FEATURE FOR EVERY WEIGHT
    #mostSimilar_feature_sorted,seeds_index = my_exp.get_most_similar_feature_from_AEweights(hid1_space,weights,orig_list_id,scene,pedestrian_cluster)


    ## FOR EACH UNIT I TAKE THE 3 MOST SIMILAR ONE AND DISPLAY IT ON AN IMAGE WITH DIFF COLORS
    #mostSimilar_feature_matrix = my_exp.get_3most_similar_feature_from_AEweights(normalized_data_HOT,weights,orig_list_id,scene,slices)


    ##Experiment: Clustering
    ####seed
    #seeds = []
    # # ##get the traj with strongest weight and use it to initialize mean shift
    #map(lambda s:seeds.append(normalized_data_HOT_pedestrian[s]),list(np.unique(seeds_index)))

    #cluster_pred_AE = my_exp.cluster_fit_predict_meanShift(normalized_data_HOT_pedestrian,seeds)



    cluster_pred_AE=load_matrix_pickle('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/AE_CHOT_clusterPrediction.txt')
    freq_counter_seedsAE =  Counter(cluster_pred_AE)

    #color_traj_based_on_clusters(pedestrian_cluster,cluster_pred_AE,freq_counter_seedsAE,scene)
    #save_matrix_pickle(cluster_pred_AE,'C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/AE_CHOT_clusterPrediction.txt')



    #abnormal_AE_HOT_classes = [0,5,11,18,24,14,12]#,25
    abnormal_AE_CHOT_classes = [4,8,15,11,17,7,22,24]

    AE_HOT_normal,AE_HOT_abnormal = normal_abnormal_class_division(freq_counter_seedsAE,cluster_pred_AE,normalized_data_HOT_pedestrian,abnormal_AE_CHOT_classes)
##----------------------------------------------------------------------------------------------------------------##
    ## Experiment: Clustering

    ##no initial seed
    #cluster_pred_raw = my_exp.cluster_fit_predict_meanShift(normalized_data_HOT_pedestrian,[])

    #cluster_pred_raw = my_exp.cluster_fit_predict_kmeans(normalized_data_HOT_pedestrian,k=26)

    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/clusterPrediction_no_seeds.txt', 'rb') as handle:
    #     cpickle.dump(cluster_pred_raw,handle)
    #freq_counter_unseed =  Counter(cluster_pred_raw)
    #print freq_counter_unseed


    ##getting same number of cluster with raw features
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
    #
    #color_traj_based_on_clusters(pedestrian_cluster,cluster_pred_raw,freq_counter_unseed,scene)

    ############

    #####TEST THE CLUSTERING
    #load cluster model
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/HOT_clusterModel.gz', 'rb') as handle:
    #     HOT_model_cluster = pickle.load(handle)
    #
    # cluster_prediction_HOT = HOT_model_cluster.predict(normalized_data_HOT_pedestrian)
    #
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/HOT_clusterPrediction.txt', 'wb') as handle:
    #     pickle.dump(cluster_prediction_HOT,handle)

    #load cluster prediction
    print 'normal HOT'
    cluster_prediction_HOT=load_matrix_pickle('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/CHOT_clusterPrediction.txt')
    freq_counter_HOT=  Counter(cluster_prediction_HOT)

    #color_traj_based_on_clusters(pedestrian_cluster,cluster_prediction_HOT,freq_counter_HOT,scene)

    #abnormal_HOT_classes = [1,3,12,16,17,20,21]#,24
    abnormal_CHOT_classes = [9,13,14,17,20,23,27,47]
    # Divide clustering in normal abnormal groups
    HOT_normal,HOT_abnormal = normal_abnormal_class_division(freq_counter_HOT,cluster_prediction_HOT,normalized_data_HOT_pedestrian,abnormal_CHOT_classes)



    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/SPEED_clusterModel.gz', 'rb') as handle:
    #     SPEED_model_cluster = pickle.load(handle)
    #
    # cluster_prediction_SPEED = SPEED_model_cluster.predict(normalized_data_SPEED_pedestrian)
    #
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/SPEED_clusterPrediction.txt', 'wb') as handle:
    #     pickle.dump(cluster_prediction_SPEED,handle)



    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/SPEED_clusterPrediction.txt', 'rb') as handle:
    #     cluster_prediction_SPEED=pickle.load(handle)
    #
    # freq_counter=  Counter(cluster_prediction_SPEED)

    # abnormal_SPEED_classes = [3,4,10,8,17,19,18,27]
    #
    # SPEED_normal_pedestrian_class=[]
    # SPEED_abnormal_pedestrian_class=[]
    #
    # for traj in freq_counter.keys()[:30]:
    #     if traj in abnormal_SPEED_classes:
    #         index = np.where(cluster_prediction_SPEED == traj)[0]
    #         for i in index:
    #             SPEED_abnormal_pedestrian_class.append(normalized_data_SPEED_pedestrian[i])
    #     else:
    #         index = np.where(cluster_prediction_SPEED == traj)[0]
    #         for i in index:
    #             SPEED_normal_pedestrian_class.append(normalized_data_SPEED_pedestrian[i])
    # print freq_counter
    # color_traj_based_on_clusters(pedestrian_cluster,cluster_prediction_SPEED,freq_counter,scene)



    hot_accuracy = []
    speed_accuracy =[]
    AE_accuracy=[]
    for ten_fold in range(0,10):
        print 'test set '+str(int(len(AE_HOT_normal)*0.1))
        test_index = random.sample(range(0,len(AE_HOT_normal)),int(len(AE_HOT_normal)*0.1))

        hot_accuracy.append(my_exp.classification_clustering_accuracy(HOT_normal,HOT_abnormal,test_index))

        AE_accuracy.append(my_exp.classification_clustering_accuracy(AE_HOT_normal,AE_HOT_abnormal,test_index))

        #speed_accuracy.append(my_exp.classification_clustering_accuracy(SPEED_normal_pedestrian_class,SPEED_abnormal_pedestrian_class,cluster_prediction_SPEED,test_index))

    print sum(hot_accuracy)/len(hot_accuracy)
    print sum(AE_accuracy)/len(AE_accuracy)
    #########


    ####A)similarity matrix in the cluster space to delete the similar seeds
    # sim_matrix_seeds= similarity.euclidean_distances(np.array(seeds))
    # print sim_matrix_seeds
    #
    # index_min = []
    # for r in sim_matrix_seeds:
    #     #get the second min of the list and get the index
    #     second_min = min(n for n in r if n!=min(r))
    #     index_min.append(np.where(r == second_min)[0][0])
    #
    #
    # print len(index_min)
    #
    # new_seeds= []
    # for i,s in enumerate(seeds):
    #     if i not in index_min:
    #         new_seeds.append(s)
    #
    # print len(new_seeds)



if __name__ == '__main__':
    main()