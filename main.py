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
from torchvision.transforms import transforms  # version 0.13.1
from PIL import Image

# Comment this out when not needing to time functions
import datetime

from LCM.Publisher import LcmPublisher 

image_directory = '/NVMEDATA/highmag/images_to_classify'
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

categorized_list = ("Aggregate", "Ciliate", "Diatom", "Dinoflagellate", "Other")

num_labels = len(categorized_list)

SAVE_IMAGES = True
DELETE_IMAGES = False


transform = transforms.Compose([ 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def init_categoried_files():
    """ Creates the Categorized files that will eventually be populated with timestamped dirs with categorized images

    Args:
        None

    Returns:
        None

    """

    current_dir = "/NVMEDATA"
    try:
        for entry in categorized_list:
            path = os.path.join(current_dir, "Categorized")
            path = os.path.join(path, entry)
            os.makedirs(path, exist_ok = True)
    except OSError as error:
        print(error)

def save_categoried_image(image, label, name, ISO_date, ISO_time):
    """Saves labeled image in specific folders

    Args:
        image: The image data
        label (int): the image category as given by the model 
        name (string): image name for proper saving
        ISO_date (string): ISO date descriptor only for directory creation.
        ISO_time (_type_): ISO time descriptor to record each categorized run
    """

    #print(f"Saving Image: {name}")

    
    path = "/NVMEDATA"
    path = os.path.join(path, "Categorized")

    # Organize the images. Wanted a switch statement but python version was too old
    label = labels_map[label]
    if "Aggregate" == label:
        path = os.path.join(path, "Aggregate")
    elif "Ciliate" == label:
        path = os.path.join(path, "Ciliate")
    elif "Diatom" in label:
        path = os.path.join(path, "Diatom")
    elif "Dinoflagellate" in label:
        path = os.path.join(path, "Dinoflagellate")
    else:
        path = os.path.join(path, "Other")
    
    # This is to debug in case their is an issue in making the directory
    try:
        path = os.path.join(path, ISO_date)
        path = os.path.join(path, ISO_time)
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print("error in making directory")

    # This is to debug in case their is an issue in making the categorized image
    try:
        path = os.path.join(path, name)
        cv2.imwrite(path+'.jpg', image)
    except OSError as error:
        print("error in making image")  


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
    #print(image_path)
    return output['image'], os.path.split(image_path)[1]

def classify_images(image_list, img_names, ISO_date, ISO_time):
    """ Using a trained model, convert images to labels and return them

    Args:
        image_list (list): List of images to classify

    Returns:
        list: list of labels
    """
    start_time = time.perf_counter()


    labels = []
    label_counts = [0,0,0,0,0]
    if image_list is not None:
        for (image, name) in zip(image_list, img_names):
            # Apply transformation and move to device
            image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = predicted.item()

            # Convert 15 labels to 5
            label = labels_map[predicted_label]
            if "Aggregate" == label:
                labels.append(0)
                label_counts[0] += 1
            elif "Ciliate" == label:
                labels.append(1)
                label_counts[1] += 1
            elif "Diatom" in label:
                labels.append(2)
                label_counts[2] += 1
            elif "Dinoflagellate" in label:
                labels.append(3)
                label_counts[3] += 1
            else:
                labels.append(4)
                label_counts[4] += 1

            

            save_categoried_image(image, predicted_label, name, ISO_date, ISO_time)

    # Print statements for testing:
    print(f"Number of {categorized_list[0]} classified: {label_counts[0]}")
    print(f"Number of {categorized_list[1]} classified: {label_counts[1]}")
    print(f"Number of {categorized_list[2]} classified: {label_counts[2]}")
    print(f"Number of {categorized_list[3]} classified: {label_counts[3]}")
    print(f"Number of {categorized_list[4]} classified: {label_counts[4]}")

    end_time = time.perf_counter()

    print (f"Classify images took: {((end_time - start_time) ):.03f}")


    return labels

def process_images(image_directory, proc_settings):
    """ load and process a directory of images, removing each image that is processed

    Args:
        image_directory (str): directory of tif images
        proc_settings (dict): processing settings

    Returns: 
        list: list of labels assigned to the images
    """
    start_time = time.perf_counter()


    image_list = sorted(glob.glob(os.path.join(image_directory, '*.tif')))
    
    # Load and process images
    processed_image_list = []
    img_name_list=[]
    for image_path in image_list:
        img, img_name = load_image(image_path, proc_settings)
        img_name_list.append(img_name)
        if img is not None:
            if DELETE_IMAGES:
                os.remove(image_path)
            processed_image_list.append(img)
            if SAVE_IMAGES:
                cv2.imwrite(image_path+'.jpg', img)
            
    # Label images
    #lables = classify_images(processed_image_list)
    end_time = time.perf_counter()
    print (f"Process images took: {((end_time - start_time) ):.03f}")


    return processed_image_list, img_name_list

def quantify_images(labels, elapsed_time=30):
    """ Convert labels into quantitative abundance estimates

    Args:
        labels (list): list of labels
        elapsed_time (int, optional): duration over which the images were collected. Defaults to 30.

    Returns:
        list: list of abundances in #/ml
    """
    start_time = time.perf_counter()
    counts = np.zeros(num_labels)
    for l in labels:
        counts[l] += 1
        
    expected_concentration = counts / (elapsed_time * cameras_fps * imaged_volume) 
    #print(expected_concentration)
    end_time = time.perf_counter()
    print (f"Quantify images took: {((end_time - start_time) ):.03f}")

    return expected_concentration

def publish_to_slate(expected_concentration, pub):
    """ publish concentration estimate to the tethys_slate channel

    Args:
        expected_concentration (list): list of concentration estimates
        pub (LcmPublisher): publisher object
    """
    # map the concentrations into a single float value (for now just pick one category)
    diatom_concentration = expected_concentration[2]
    dinoflagellate_concentration = expected_concentration[3]
    
    # publish to the slate
    pub.msg.epochMillisec = int(time.time() * 1000)
    pub.add_float('._planktivore_diatoms', unit='n/a', val=np.float32(diatom_concentration))
    pub.add_float('._planktivore_dinoflagellates', unit='n/a', val=np.float32(dinoflagellate_concentration))

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
    model = torch.load('HM_model.pth')
    model.eval()  # Set the model to evaluation mode
    torch.no_grad()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # create inital categorized files
    init_categoried_files()
    
    #while True:
    current_time = time.time()

    # Used for categorized files. Time is taken here to prevented every image getting it's own directory
    ISO_time = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    ISO_date, ISO_time = ISO_time.split('T')

    images, img_names = process_images(image_directory, proc_settings)
    
    print(f"Processed {len(images} images.")

    labels = classify_images(images, img_names, ISO_date, ISO_time)
    quants = quantify_images(labels)
    publish_to_slate(quants, pub)
    elapsed_time = time.time() - current_time
    if elapsed_time < processing_interval:
        print('sleeping for ' + str(processing_interval - elapsed_time) + ' seconds')
        time.sleep(processing_interval - elapsed_time)
        
