import numpy as np
from sklearn.cluster import KMeans,MeanShift
from sklearn.manifold import Isomap
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import random
from collections import Counter
import cPickle


import main as main_camera017

__logistic = LogisticRegression()
__my_svm = svm.SVC()


def cluster_fit_predict_meanShift(data,seeds):

    if len(seeds)>0:
        my_meanShift = MeanShift(bandwidth=0.5,bin_seeding=True,n_jobs=-1,seeds=seeds)
    else:
        my_meanShift = MeanShift(bandwidth=0.5,bin_seeding=True,n_jobs=-1)

    #trained_cluster = my_meanShift.fit(data)
    #cluster_prediction = np.array(trained_cluster.predict(data))
    my_meanShift.fit(data)
    #cluster_prediction = np.array(my_meanShift.fit_predict(data))
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/CHOT_clusterModel.gz', 'wb') as handle:
    #     cPickle.dump(my_meanShift,handle,protocol=2)

    cluster_prediction = np.array(my_meanShift.predict(data))
    return cluster_prediction


def cluster_fit_predict_kmeans(data,k):
    ##KMeans
    my_kmean = KMeans(n_clusters=k,init='k-means++',n_jobs=-1)
    cluster_prediction = np.array(my_kmean.fit_predict(data))
    return cluster_prediction


def data_reduction_for_visualization(cluster_center):

    manifold= Isomap()
    #manifold = LocallyLinearEmbedding()

    centers_red_dim =  manifold.fit_transform(np.array(cluster_center))
    print centers_red_dim
    plt.scatter(centers_red_dim[:,0],centers_red_dim[:,1])
    plt.show()


def get_most_similar_feature_from_AEweights(normalized_data_HOT,weights,orig_list_id,scene,slices):


    mostSimilar_feature_arr = np.zeros((len(weights),2))


    for i,weight in enumerate(weights):
        diff = []
        diff_append = diff.append


        map(lambda feature: diff_append(abs(sum(feature-weight))), normalized_data_HOT)
        min_diff = min(diff)

        ##store the minimum of the differences and its index to recover the trajectory it belongs to
        mostSimilar_feature_arr[i,0] = diff.index(min_diff)
        mostSimilar_feature_arr[i,1] = min_diff


    ###sorting array according to value in second column
    mostSimilar_feature_sorted = np.array(sorted(mostSimilar_feature_arr,key=lambda x: x[1]))

    # mostSimilar_feature = mostSimilar_feature_arr[::-1]
    print mostSimilar_feature_sorted

    ###take id of strongest features in AE and show


    mostSimilar_feature = map(int,mostSimilar_feature_sorted[:,0])

    #print Counter(mostSimilar_feature)
    seeds_index = []

    cm_subsection = np.linspace(0.0, 1.0, len(mostSimilar_feature))
    colors = [ plt.cm.jet(x) for x in cm_subsection ]
    implot = plt.imshow(scene)

    for counter,unit in enumerate(mostSimilar_feature):#[:60]
        temp_img = scene.copy()
        # print unit
        traj_id = orig_list_id[unit]

        for i,s in enumerate(slices):
            if s[0].split(' ')[0] == traj_id:
                seeds_index.append(i)
                for t in s:
                    x = t.split(' ')[2]
                    y = t.split(' ')[3]
                    plt.scatter(int(x),int(y),color=colors[counter],s=5)

    plt.show()
    return mostSimilar_feature_sorted,seeds_index


def get_3most_similar_feature_from_AEweights(normalized_data_HOT,weights,orig_list_id,scene,slices):
    mostSimilar_feature_matrix = np.zeros((len(weights)*3,normalized_data_HOT.shape[0]))

    cm_subsection = np.linspace(0.0, 1.0, len(weights))
    colors = [ plt.cm.jet(x) for x in cm_subsection ]

    implot = plt.imshow(scene)

    for i,weight in enumerate(weights):
        diff = []
        diff_append = diff.append

        map(lambda feature: diff_append(abs(sum(feature-weight))), normalized_data_HOT)

        #mostSimilar_feature_matrix[i,:] = diff
        bests_f = np.sort(np.unique(diff))[:3]
        print bests_f

        index_bests_f = np.zeros((3))
        for h,bf in enumerate(bests_f):
            index_bests_f[h] = list(diff).index(bf)

        index_bests_f = map(int,index_bests_f)

        mostSimilar_feature_matrix[i]=normalized_data_HOT[index_bests_f[0]]
        mostSimilar_feature_matrix[i+1]=normalized_data_HOT[index_bests_f[1]]
        mostSimilar_feature_matrix[i+2]=normalized_data_HOT[index_bests_f[2]]


        for unit in index_bests_f:
            #temp_img = scene.copy()
            #print unit[0]

            traj_id = orig_list_id[unit]

            for s in slices:
                if s[0].split(' ')[0] == traj_id:
                    #seeds_index.append(i)
                    for t in s:
                        x = t.split(' ')[2]
                        y = t.split(' ')[3]
                        plt.scatter(int(x),int(y),color=colors[i],s=5)

    plt.show()

    return mostSimilar_feature_matrix


def classification_clustering_accuracy(normal_data_pedestrian,abnormal_data_pedestrian,test_index):
    ####SUPERVISED CLASSIFIER

    all_data= normal_data_pedestrian+abnormal_data_pedestrian
    all_data_labels=np.vstack((np.ones((len(normal_data_pedestrian),1)),np.zeros((len(abnormal_data_pedestrian),1))))

    training_samples = []
    training_labels=[]
    test_samples = []
    test_labels=[]

    for i in range(0,len(all_data)):
        if i in test_index:
            test_samples.append(all_data[i])
            test_labels.append(all_data_labels[i])

        else:
            training_samples.append(all_data[i])
            training_labels.append(all_data_labels[i])


    #train Logistic regression classifier
    logistic_regression_train(training_samples,training_labels)

    #test
    pred = logistic_regression_predict(test_samples)

    counter = 0

    for i in range(0,len(pred)):
        if pred[i]==test_labels[i]:
            counter+=1

    return float(counter)/len(test_labels)


def cars_vs_pedestrian(normalized_data,slices,scene):
    cluster_pred = cluster_fit_predict_kmeans(normalized_data,k=3)

    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/AE_parameters/30000id/30000_clusterPrediction_no_seeds.txt', 'rb') as handle:
    #     cluster_pred_raw=cpickle.load(handle)

    freq_counter =  Counter(cluster_pred)

    #main_camera017.color_traj_based_on_clusters(slices,cluster_pred,freq_counter,scene)

    clusters = []
    for trj_id in freq_counter.keys():
        temp_clusters = []
        ##get index of every class
        index = np.where(cluster_pred == trj_id)[0]

        for i in index:
            temp_clusters.append(slices[i])

        clusters.append(temp_clusters)


    if len(clusters[0])>len(clusters[1]) and len(clusters[0]) > len(clusters[2]):
        pedestrian_cluster = clusters[0]
        cars_cluster = clusters[1]
        cars_cluster = cars_cluster+clusters[2]
        #cars_cluster = np.vstack((cars_cluster,clusters[2]))
    elif len(clusters[1])>len(clusters[0]) and len(clusters[1])>len(clusters[2]):
        pedestrian_cluster = clusters[1]
        cars_cluster = clusters[0]
        cars_cluster = cars_cluster+clusters[2]
    else:
        pedestrian_cluster = clusters[2]
        cars_cluster = clusters[0]
        #cars_cluster = cars_cluster+clusters[1]

    #
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/RBH_pedestrian_matrix.txt', 'wb') as handle:
    #     cPickle.dump(pedestrian_cluster,handle,protocol=2)
    #
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/RBH_cars_matrix.txt', 'wb') as handle:
    #     cPickle.dump(cars_cluster,handle,protocol=2)
    # return False

    # implot = plt.imshow(scene)
    # for row in pedestrian_cluster[:100]:
    #     x_f,y_f,ids = get_coordinate_points(row)
    #
    #     for p in xrange(0,len(x_f)):
    #         plt.scatter(int(x_f[p]),int(y_f[p]),color='blue',s=5)
    #
    # plt.show()
    return pedestrian_cluster,cars_cluster


def logistic_regression_train(data,labels):
    __logistic.fit(data, labels)


def logistic_regression_predict(test_data):
    return __logistic.predict(test_data)


def svm_train(data,labels):
    __my_svm.fit(data,labels)


def svm_predict(test_data):
    return __my_svm.predict(test_data)