import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
tfds.disable_progress_bar()
import tensorflow_hub as hub

import warnings
warnings.filterwarnings('ignore')
import argparse
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import json
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# Import TensorFlow 


parser=argparse.ArgumentParser(description="Image Classifer -Prediction Part")
parser.add_argument("--input",default="./test_images/hard-leaved_pocket_orchid.jpg",action="store",type=str,help="image path")
parser.add_argument("--model",default="./classifer.h5",action="store",type=str,help="checkpoint file path/name")
parser.add_argument("--top_k",default=5,dest="top_k",action="store",type=int,help="return top k most")
parser.add_argument("--category_names",dest="category_names",action="store",default="label_map.json",help="mappig the category")

arg_parser=parser.parse_args()
image_path=arg_parser.input
model_path=arg_parser.model
topk=arg_parser.top_k
category_names=arg_parser.category_names



def process_image(image):
    image=tf.convert_to_tensor(image,dtype=tf.float32)
    Image=tf.image.resize(image,(224,224))
    Image=Image/255
    Image=Image.numpy()
    
    return Image

def predict(image_path,model,top_k):
    Img=Image.open(image_path)
    test_image = np.asarray(Img)
    normlized_image=process_image(test_image)
    redim_Img=np.expand_dims(normlized_image,axis=0)
    pred=model.predict(redim_Img)
    pred=pred.tolist()
    probs ,classes= tf.math.top_k(pred,k=top_k)
    probs=probs.numpy().tolist()[0]
    classes=classes.numpy().tolist()[0]
    return probs ,classes
    ##print(probs)
    ###print(classes)
   
if __name__=="__main__":
   print("start prediction ..." )
   with open(category_names,'r') as f:
        class_names=json.load(f)
        
   model_2=tf.keras.models.load_model('./model.h5',custom_objects={'KerasLayer':hub.KerasLayer})
   probs, classes = predict(image_path, model_2, topk)
   labels=[class_names[str(int(labell+1))]for labell in classes ]

   print('probs:: ', probs)
   print('classes:: ',classes)
   print('lebel:: ',labels)
   print('end prdiction ') 