import os
import cv2
import sys
import lcm
import glob
import json
import time
import random
import libs.cvtools as cvtools


import numpy as np

from LCM.Publisher import LcmPublisher 

image_directory = '/NVMEDATA/images_to_classify'
processing_interval = 30
cameras_fps = 10 # frames per second
imaged_volume = 0.0001 # volume in ml
num_labels = 14

SAVE_IMAGES = False

def load_image(image_path, proc_settings):
    """ loads a raw tif image from disk and converts to a processed image

    Args:
        image_path (str): full path to the image
        proc_settings (dict): settings to control processing

    Returns:
        array: processed image
    """
    img = cvtools.import_image(os.path.split(image_path)[0], os.path.split(image_path)[1], proc_settings)
    img_c_8bit = cvtools.convert_to_8bit(img, proc_settings)
    output = cvtools.extract_features(
            img_c_8bit,
            img,
            proc_settings, 
            save_to_disk=False,
        )
    print(image_path)
    return output['image']

def classify_images(image_list):
    
    labels = []
    if image_list is not None:
        for image in image_list:
            labels.append(random.randint(0,num_labels-1))
            
    return labels

def process_images(image_directory, proc_settings):
    """ load and process a directory of images, removing each image that is processed

    Args:
        image_directory (str): directory of tif images
        proc_settings (dict): processing settings

    Returns:
        list: list of labels assigned to the images
    """
    image_list = sorted(glob.glob(os.path.join(image_directory, '*.tif')))
    
    # Load and process images
    processed_image_list = []
    for image_path in image_list:
        img = load_image(image_path, proc_settings)
        if img is not None:
            os.remove(image_path)
            processed_image_list.append(img)
            if SAVE_IMAGES:
                cv2.imwrite(image_path+'.jpg', img)
            
    # Label images
    lables = classify_images(processed_image_list)
    
    return lables

def quantify_images(labels, elapsed_time=30):
    
    counts = np.zeros(num_labels)
    for l in labels:
        counts[l] += 1
        
    expected_concentration = counts / (elapsed_time * cameras_fps * imaged_volume)
    
    return expected_concentration

def publish_to_slate(expected_concentration, pub):
    
    # lcm stuff here
    pass    


if __name__=="__main__":

    # public channels 
    lcm_url = 'udpm://239.255.76.67:7667?ttl=1'
    
    # instantiate lcm and publisher
    lc = lcm.LCM(lcm_url)
    pub = LcmPublisher(lc)
    
    # load settings
    with open('ptvr_proc_settings.json') as file:
        proc_settings = json.load(file)
    
    while True:
        start_time = time.time()
        images = process_images(image_directory, proc_settings)
        labels = classify_images(images)
        quants = quantify_images(labels)
        publish_to_slate(quants, pub)
        print(time.time()-start_time)
        print('sleeping...')
        time.sleep(processing_interval)
        