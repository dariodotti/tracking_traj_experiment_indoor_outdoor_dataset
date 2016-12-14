import cv2
import pickle
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from os import listdir
from os.path import isfile, join

import img_proc as my_img_proc
import main as main_camera017

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
    keys = sorted(keys,key=lambda x: x)
    print len(keys)

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
            print k
            temp_track = []
            temp_track_append= temp_track.append

            map(lambda i: temp_track_append(file_content[i]) if ids[i] == k else False,xrange(len(file_content)))
            tracklets_append(temp_track)


    return tracklets

def get_coordinate_points(occurance):

    frames =map(lambda line: str(line.split(' ')[0]),occurance)

    center_xs = map(lambda line: int(float(line.split(' ')[1])),occurance)
    center_ys = map(lambda line: int(float(line.split(' ')[2])),occurance)

    bb_width = map(lambda line: int(float(line.split(' ')[3])),occurance)
    bb_height = map(lambda line: int(float(line.split(' ')[4])),occurance)



    #list_points = []
    #list_points_append = list_points.append
    #map(lambda c: list_points_append((xs[c],ys[c])),xrange(0,len(xs)))
    #apply filter to cancel noise
    #x_f,y_f =my_img_proc.median_filter(list_points)

    return frames,center_xs,center_ys,bb_width,bb_height

def main():
    ##divide image into patches(polygons) and get the positions of each one
    scene = cv2.imread('C:/Users/dario.dotti/Documents/LOST_dataset/camera017.jpg')
    list_poly = my_img_proc.divide_image(scene)

    mypath= 'C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/training/blobs/'
    only_files=[f for f in listdir(mypath) if isfile(join(mypath, f))]

    for f in only_files:
        day = f.split('_')[0]
        month = f.split('_')[1]
        my_file= ''.join([mypath,f])

        slices=read_data_tracklets(my_file,0)
        with open('C:/Users/dario.dotti/Documents/LOST_dataset/8_2013-12_2012_camera001/pedestrian_cars/training/blobs_forTraining/blobs_org_by_frames_'+day+'_'+month+'.txt', 'wb') as handle:
            pickle.dump(slices,handle)

    return False





    with open('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/pedestrian_cars/classification/blobs_org_by_frames_7_2.txt', 'rb') as handle:
        slices = pickle.load(handle)


    for n,slice in enumerate(slices):
        temp_img = scene.copy()
        frames,center_xs,center_ys,bb_width,bb_height = get_coordinate_points(slice)

        for i in range(0,len(center_ys)):
            if bb_width[i] > 15 or bb_height[i] >15:
                vertex_1 = (center_xs[i]-(bb_width[i]/2)),(center_ys[i]-(bb_height[i]/2))
                vertex_2 = (center_xs[i]+(bb_width[i]/2)),(center_ys[i]+(bb_height[i]/2))

                cv2.rectangle(temp_img,vertex_1,vertex_2,0,1)

        cv2.putText(temp_img,frames[0],(30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, 0)
        cv2.imshow('ciao',temp_img)
        cv2.waitKey(0)



if __name__ == '__main__':
    main()