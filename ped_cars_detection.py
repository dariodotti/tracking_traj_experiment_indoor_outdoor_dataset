import matplotlib.pyplot as plt
import cv2
import pickle
import cPickle
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from os import listdir
from os.path import isfile, join

import img_proc as my_img_proc
import main as main_camera017
import blobs as my_blobs
import experiments as my_exp


def get_coordinate_points(occurance):
    frames = map(lambda line: int(float(line.split(' ')[1])),occurance)
    xs = map(lambda line: int(float(line.split(' ')[2])),occurance)
    ys = map(lambda line: int(float(line.split(' ')[3])),occurance)
    ids =map(lambda line: str(line.split(' ')[0]),occurance)


    #list_points = []
    #list_points_append = list_points.append
    #map(lambda c: list_points_append((xs[c],ys[c])),xrange(0,len(xs)))
    #apply filter to cancel noise
    #x_f,y_f =my_img_proc.median_filter(list_points)

    return xs,ys,ids,frames


def display_objects(temp_img,center_xs,center_ys,bb_width,bb_height,x_traj,y_traj):

    #draw closest biggest blob
    vertex_1 = (center_xs-(bb_width/2)),(center_ys-(bb_height/2))
    vertex_2 = (center_xs+(bb_width/2)),(center_ys+(bb_height/2))
    cv2.rectangle(temp_img,vertex_1,vertex_2,0,1)

    #decide whether this blob is a car or a pedestrian
    if float(bb_width)/float(bb_height)>0.8 and bb_width > 20:
        cv2.putText(temp_img,'CAR',(30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, 0)
    else:
        cv2.putText(temp_img,'PEDESTRIAN',(30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, 0)



    implot = plt.imshow(temp_img)
    for p in xrange(0,len(x_traj)):
                plt.scatter(int(x_traj[p]),int(y_traj[p]),color='blue',s=5)


    plt.show()

def supervised_training_fromClustering():
    with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/CHOT_pedestrian_matrix.txt', 'rb') as handle:
        pedestrian_matrix = np.array(pickle.load(handle))
    with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/CHOT_cars_matrix.txt', 'rb') as handle:
        cars_matrix = np.array(pickle.load(handle))

    # pedestrian_matrix_SPEED = main_camera017.velo_acc_curva_feature(list_poly,pedestrian_matrix)
    # cars_matrix_SPEED = main_camera017.velo_acc_curva_feature(list_poly,cars_matrix)

    print pedestrian_matrix.shape,cars_matrix.shape

    training_matrix = np.vstack((pedestrian_matrix,cars_matrix))
    training_labels= np.vstack((np.ones((pedestrian_matrix.shape[0],1)),np.zeros((cars_matrix.shape[0],1))))

    my_exp.logistic_regression_train(training_matrix,training_labels)
    return True

def separate_cars_ped_using_blobs():

    mypath='C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/training/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    cars_matrix = []
    ped_matrix = []

    for f in onlyfiles:
        print f
        #get trajectories
        tracklets=main_camera017.read_data_tracklets(mypath+f,0)

        #get blobs for the same say
        blob_file = 'C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/training/blobs_forTraining/'+'blobs_org_by_frames_'+f
        with open(blob_file,'rb') as handle:
            blobs = pickle.load(handle)


        for n,trak in enumerate(tracklets):
            x_traj,y_traj,ids,frames_traj = get_coordinate_points(trak)

            ##get the frame where the traj ends
            reference_p_traj = [frames_traj[-1],x_traj[-1],y_traj[-1]]

            ##take the blobs in the same frame of the trajectory
            for b in range(0,len(blobs)):

                if int(blobs[b][0].split(' ')[0]) == reference_p_traj[0]:
                    #print reference_p_traj[0]
                    #print blobs[reference_p_traj[0]]
                    break
            frames_blob,center_xs,center_ys,bb_width,bb_height = my_blobs.get_coordinate_points(blobs[b])

            index_closeBlobs=[]
            ##check the closest blob to the selected trajectory
            for i in range(0,len(frames_blob)):
                distance = int(np.sqrt((np.power((reference_p_traj[1]-center_xs[i]),2)+np.power((reference_p_traj[2]- center_ys[i]),2))))
                if distance < 60:
                    #save the area of each blobs
                    index_closeBlobs.append([i,int(bb_width[i])*int(bb_height[i])])

            ###sorting array according to value in second column (areas
            index_closeBlobs = np.array(sorted(index_closeBlobs,key=lambda x: x[1]))
            index_closeBlobs = index_closeBlobs[::-1]

            if len(index_closeBlobs) >1:

                #decide whether this blob is a car or a pedestrian
                if float(bb_width[index_closeBlobs[0][0]])/float(bb_height[index_closeBlobs[0][0]])>0.8 and bb_width[index_closeBlobs[0][0]] > 20:
                    #matrix for object CAR
                    cars_matrix.append(trak)
                else:
                    #matrix for object PEDESTRIAN
                    ped_matrix.append(trak)

    with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/cars_traj.txt','wb') as handle:
        cPickle.dump(cars_matrix,handle,protocol=2)

    with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/ped_traj.txt','wb') as handle:
        cPickle.dump(ped_matrix,handle,protocol=2)

def supervised_training_withoutClustering(list_poly,scene):
    with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/cars_traj.txt','rb') as handle:
        cars_matrix = cPickle.load(handle)

    with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/ped_traj.txt','rb') as handle:
        ped_matrix=cPickle.load(handle)


    # for raw_tra in ped_matrix[:300]:
    #     x_f,y_f,ids = main_camera017.get_coordinate_points(raw_tra)
    #
    #     # #draw filtered points
    #     for p in range(0,len(x_f)):
    #             cv2.circle(scene,(int(x_f[p]),int(y_f[p])),2,255,-1)#my_room
    #
    # for raw_tra in cars_matrix[:300]:
    #     x_f,y_f,ids = main_camera017.get_coordinate_points(raw_tra)
    #
    #     # #draw filtered points
    #     for p in range(0,len(x_f)):
    #             cv2.circle(scene,(int(x_f[p]),int(y_f[p])),2,(0,0,255),-1)#my_room
    #
    #
    # cv2.imshow('camera001',scene)
    # cv2.waitKey(0)



    # SPEED_ped = main_camera017.velo_acc_curva_feature(list_poly,ped_matrix)
    # SPEED_car = main_camera017.velo_acc_curva_feature(list_poly,cars_matrix)


    HOT_ped,id_list = main_camera017.histograms_of_oriented_trajectories(list_poly,ped_matrix)
    HOT_car,id_list = main_camera017.histograms_of_oriented_trajectories(list_poly,cars_matrix)




    training_matrix = np.vstack((HOT_ped,HOT_car))
    training_labels= np.vstack((np.ones((HOT_ped.shape[0],1)),np.zeros((HOT_car.shape[0],1))))

    my_exp.logistic_regression_train(training_matrix,training_labels)
    return True

def supervised_test(normalized_data):

    return my_exp.logistic_regression_predict(normalized_data)


def main():
    scene = cv2.imread('C:/Users/dario.dotti/Documents/LOST_dataset/camera001.jpg')
    list_poly = my_img_proc.divide_image(scene)

    ##pedestrian cars without clustering
    #separate_cars_ped_using_blobs()

    trained=supervised_training_withoutClustering(list_poly,scene)
    if trained: print 'logistic regression trained'


    ##pedestrian cars with clustering
    #trained =supervised_training_fromClustering()
    #if trained: print 'logistic regression trained'

    ######test##################

    with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/classification/blobs_org_by_frames_19_12.txt', 'rb') as handle:
        blobs = pickle.load(handle)


    tracklets=main_camera017.read_data_tracklets('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/single_files/19_12.txt',0)

    tracklets = tracklets[:50]


    #normalized_data_SPEED = main_camera017.velo_acc_curva_feature(list_poly,tracklets)
    normalized_data_HOT,id_list = main_camera017.histograms_of_oriented_trajectories(list_poly,tracklets)


    predictions = supervised_test(normalized_data_HOT)

    counter = 0
    for n,trak in enumerate(tracklets):

        temp_img = scene.copy()


        x_traj,y_traj,ids,frames_traj = get_coordinate_points(trak)

        ##get the frame where the traj ends
        reference_p_traj = [frames_traj[-1],x_traj[-1],y_traj[-1]]

        ##take the blobs in the same frame of the trajectory
        for b in range(0,len(blobs)):

            if int(blobs[b][0].split(' ')[0]) == reference_p_traj[0]:
                #print reference_p_traj[0]
                #print blobs[reference_p_traj[0]]
                break

        frames_blob,center_xs,center_ys,bb_width,bb_height = my_blobs.get_coordinate_points(blobs[b])

        index_closeBlobs=[]
        ##check the closest blob to the selected trajectory
        for i in range(0,len(frames_blob)):
            distance = int(np.sqrt((np.power((reference_p_traj[1]-center_xs[i]),2)+np.power((reference_p_traj[2]- center_ys[i]),2))))
            if distance < 60:
                #save the area of each blobs
                index_closeBlobs.append([i,int(bb_width[i])*int(bb_height[i])])

        ###sorting array according to value in second column (areas
        index_closeBlobs = np.array(sorted(index_closeBlobs,key=lambda x: x[1]))
        index_closeBlobs = index_closeBlobs[::-1]



        #decide whether this blob is a car or a pedestrian
        if float(bb_width[index_closeBlobs[0][0]])/float(bb_height[index_closeBlobs[0][0]])>0.8 and bb_width[index_closeBlobs[0][0]] > 20:
            #label for object CAR
            gt = 0
        else:
            #label for object PEDESTRIAN
            gt=1

        if predictions[n]== gt:
            counter = counter+1

        #display_objects(temp_img,center_xs[index_closeBlobs[0][0]],center_ys[index_closeBlobs[0][0]],bb_width[index_closeBlobs[0][0]],bb_height[index_closeBlobs[0][0]],x_traj,y_traj)
    print 'clssification accuracy'
    print float(counter)/float(len(tracklets))








if __name__ == '__main__':
    main()