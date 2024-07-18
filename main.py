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
import torch
#import torchvision.models as models 
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd

from LCM.Publisher import LcmPublisher 

image_directory = '/NVMEDATA/images_to_classify'
processing_interval = 30
cameras_fps = 10 # frames per second
imaged_volume = 0.0001 # volume in ml

labels_map = {
    0: "Aggregate",
    1: "Bad_Mask",
    2: "Blurry",
    3: "Camera_Ring",
    4: "Ciliate",
    5: "Copepod",
    6: "Diatom:_Long_Chain",
    7: "Diatom:_Long_Single",
    8: "Diatom:_Spike_Chain",
    9: "Diatom:_Sprial_Chain",
    10: "Diatom:_Square_Single",
    11: "Dinoflagellate:_Circles",
    12: "Dinoflagellate:_Horns",
    13: "Phaeocystis",
    14: "Radiolaria"
}

num_labels = len(labels_map)

SAVE_IMAGES = True
DELETE_IMAGES = False


transform = transforms.Compose([ 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# def create_image_csv():
#     print("Saving CSV")

#     folder_path = '.'
#     data = {
#         "image": [],
#         "label": []
#     }
#     #loop through training_data to see all the training dirs, then loop through the training dirs to get all the images. Then save image file name and it's file label in a csv.
#     for dirs in os.listdir(folder_path):
#         working_dir = os.path.join(folder_path, dirs)
#         for files in os.listdir(working_dir):
#             data["image"].append(files)
#             data["label"].append(dirs)

#     df = pd.DataFrame(data)
#     # This creates the csv name: ISO date format
#     filename = time.strftime("%Y-%m-%dT%H:%M.csv", time.gmtime())
#     directory_name=time.strftime("%Y-%m-%d", time.gmtime())

#     # Try to save the CSV, If you can't, error out.
#     try:
#         path = os.path.join("History", "CSVs")
#         path = os.path.join(path, directory_name)
#         os.makedirs(path, exist_ok = TrueSAVE_IMAGES)
#         df.to_csv(os.path.join(path, filename), header=False, index=False)
#     except OSError as error:
#         print(error)

def load_image(image_path, proc_settings):
    """ loads a raw tif image from disk and convertSAVE_IMAGESs to a processed image

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
    """ Using a trained model, convert images to labels and return them

    Args:
        image_list (list): List of images to classify

    Returns:
        list: list of labels
    """
    ###TODO###
    labels = []
    if image_list is not None:
        for image in image_list:
             # Apply transformation and move to device
            #print(type(image))
            if isinstance(image, int):
                print(image)
            else:
                print ("I am categoring")
                image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_label = predicted.item()
                #predicted_label = labels_map[predicted_label]
                labels.append(predicted_label)
                #labels.append(random.randint(0,num_labels-1))
            
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
            if DELETE_IMAGES:
                os.remove(image_path)
            processed_image_list.append(img)
            if SAVE_IMAGES:
                cv2.imwrite(image_path+'.jpg', img)
            
    # Label images
    lables = classify_images(processed_image_list)
    
    return lables

def quantify_images(labels, elapsed_time=30):
    """ Convert labels into quantitative abundance estimates

    Args:
        labels (list): list of labels
        elapsed_time (int, optional): duration over which the images were collected. Defaults to 30.

    Returns:
        list: list of abundances in #/ml
    """
    
    counts = np.zeros(num_labels)
    for l in labels:
        counts[l] += 1
        
    expected_concentration = counts / (elapsed_time * cameras_fps * imaged_volume)
    print(expected_concentration)
    
    return expected_concentration

def publish_to_slate(expected_concentration, pub):
    """ publish concentration estimate to the tethys_slate channel

    Args:
        expected_concentration (list): list of concentration estimates
        pub (LcmPublisher): publisher object
    """
    # map the concentrations into a single float value (for now just pick one category)
    diatom_concentration = expected_concentration[3]
    
    # publish to the slate
    pub.msg.epochMillisec = int(time.time() * 1000)
    pub.add_float('._planktivore_diatoms', unit='n/a', val=np.float32(diatom_concentration))
    pub.publish('tethys_slate')
    


if __name__=="__main__":

    # public channels 
    lcm_url = 'udpm://239.255.76.67:7667?ttl=1'
    
    # instantiate lcm and publisher
    lc = lcm.LCM(lcm_url)
    pub = LcmPublisher(lc)
    
    # load settings
    with open('ptvr_proc_settings.json') as file:
        proc_settings = json.load(file)

    # load machine learning model
    #model = models.resnet18(weights='DEFAULT')
    model = torch.load('HM_model.pth')
    model.eval()  # Set the model to evaluation mode
    torch.no_grad()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    #while True:
    current_time = time.time()
    labels = process_images(image_directory, proc_settings)
    #labels = classify_images(images)
    quants = quantify_images(labels)
    publish_to_slate(quants, pub)
    elapsed_time = time.time() - current_time
    if elapsed_time < processing_interval:
        print('sleeping for ' + str(processing_interval - elapsed_time) + ' seconds')
        time.sleep(processing_interval - elapsed_time)
        