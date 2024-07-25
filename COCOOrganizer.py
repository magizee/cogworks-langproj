from cogworks_data.language import get_data_path
import pickle

from pathlib import Path
import json

class COCOOrganizer:
    def __init__(self):
        self.load_data()
        # store all img + caption IDs
        self.image_IDs = []
        self.caption_IDs = []
        # Initialize Various mappings between image/caption IDs, and associating caption-IDs with captions        
        self.image_to_caption_IDs = {}
        self.caption_to_image = {}
        self.caption_ID_to_caption = {}
        self.caption_ID_to_feature = {}
        
        self.organize()
        
    def load_data(self):
        # load COCO data
        filename = get_data_path("captions_train2014.json")
        with Path(filename).open() as f:
            self.COCO_data = json.load(f)
        # load image features
        with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
            self.resnet18_features = pickle.load(f)
    
    def organize(self):
        #loop thru images
        for image_id in self.resnet18_features["images"]:
            image_ID = image_id        
            #store each image ID in one big list
            self.image_IDs.append(image_ID)
            #init image-> captions dictionary
            self.image_to_caption_IDs[image_ID] = []
        for annotation in self.COCO_data["annotations"]:
            #get IDs and caption for each annotation
            caption_ID = annotation['id'] 
            image_ID = annotation['image_id']
            caption = annotation['caption']
            #add info to lists + dictionaries
            self.caption_IDs.append(caption_ID)
            self.caption_to_image[caption_ID] = image_ID
            self.caption_ID_to_caption[caption_ID] = caption
            self.image_to_caption_IDs[image_ID].append(caption_ID)
            self.caption_ID_to_feature[caption_ID] = self.resnet18_features[image_ID]

    def get_caption_ID(self, image_ID):
        return self.image_to_caption_IDs.get(image_ID)
    def get_image_ID(self, caption_ID):
        return self.caption_to_image.get(caption_ID)
    def get_caption(self, caption_ID):
        return self.caption_ID_to_caption.get(caption_ID)