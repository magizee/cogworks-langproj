from cogworks_data.language import get_data_path

from pathlib import Path
import json



class COCOOrganizer:
    def __init__(self):
        # load COCO data
        filename = get_data_path("captions_train2014.json")
        with Path(filename).open() as f:
            self.COCO_data = json.load(f)
        # store all img + caption IDs
        self.imageIDs = [image["id"] for image in self.COCO_data["images"]]
        self.caption_IDs = [caption['id'] for caption in self.COCO_data["annotations"]]
        # Initialize Various mappings between image/caption IDs, and associating caption-IDs with captions        

    def get_caption_IDs(self, img_ID):
        return None
    def get_image_ID(self, caption_ID):
        return None
    def get_caption(self, caption_ID):
        return None


    
    





    
    
        
        
