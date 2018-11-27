import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import json
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/zhuo/work/toolbox/Mask_RCNN/")

print (ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco"))  # To find local version
import coco

get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.3

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



# save objectness results to binary masks

path = '/home/zhuo/work/dataset/CVPR2016_VOS_Benchmark/davis/JPEGImages/480p'
dir_save = 'results/objmask'

video_list = os.listdir(path)
video_list.sort()
for k in range(len(video_list)):    
    video = video_list[k]
    frame_list = os.listdir(os.path.join(path, video))
    frame_list.sort()
    
    dir_save_name = os.path.join(dir_save, video)
    if not os.path.exists(dir_save_name):
        os.makedirs(dir_save_name)
    
    
    time_all = 0;    

    for frame_id in range(len(frame_list)):
        frame = skimage.io.imread(os.path.join(path, video, '{:05d}.jpg'.format(frame_id)))
        # object proposals
        
        start_time = time.time()
        results = model.detect([frame], verbose=0)
        result = results[0]
        
        obj_mask = np.zeros((frame.shape[0], frame.shape[1])).astype(np.bool)
        for i in range(result['masks'].shape[2]):
            obj_mask = np.logical_or(obj_mask, result['masks'][:,:,i])
        elapsed_time = time.time() - start_time
        time_all =  time_all + elapsed_time
        
        file_save_name = os.path.join(dir_save_name, '{:05d}.jpg'.format(frame_id))
        skimage.io.imsave(file_save_name, obj_mask.astype(np.uint8)*255)
        
        print ('Progress: video = ', k, '/', len(video_list), ', frame = ', frame_id, '/', len(frame_list))
            
    avg_time = time_all/len(frame_list)
    
    print (avg_time)






