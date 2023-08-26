# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy.core.fromnumeric import shape
from model.model import DynamicIM2TEXT
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from functools import partial
from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms
import sys
import pdb
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
import pickle
import cv2

from utils import is_master

# for testing purposes
retrieved_items_json_arr = []


def prepare_img(img_file, transform):
    return transform(Image.open(img_file))

def visualize_results(model, img2text, args, prompt, dataloader):        
    model.eval()
    img2text.eval()   
    if not os.path.exists(args.demo_out):
        os.makedirs(args.demo_out)        
    if not os.path.exists(os.path.join(args.demo_out, "images")):
        os.makedirs(os.path.join(args.demo_out, "images"))
    text = []
    id_split = tokenize(["*"])[0][1]
    for p in prompt:
        text_tokens = tokenize(p)
        text.append(text_tokens)
        assert id_split in text_tokens
    text = torch.cat(text, dim=0)    
    text = text.cuda(args.gpu, non_blocking=True)    
    all_image_features, all_image_filenames = [], []
    m = model.module if args.distributed or args.dp else model
    query_file = args.query_file
    path_save = os.path.join("./data", args.retrieval_data.split('/')[-1].split('.')[0]+".pkl")
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        all_image_features = data['feats']
        all_image_filenames = data['path']
        all_image_features = torch.from_numpy(all_image_features).cuda(args.gpu, non_blocking=True)
    else:
        ## Extract features of target images. 
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, filenames = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                image_features = m.encode_image(images)           
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                all_image_features.append(image_features)
                for name in filenames:
                    all_image_filenames.append(name)
            all_image_features = torch.cat(all_image_features, dim=0)
            dict_save = {}
            dict_save['feats'] = all_image_features.data.cpu().numpy()
            dict_save['path'] = all_image_filenames
            with open(path_save,"wb") as f:
                pickle.dump(dict_save,f)
    f = open(os.path.join(args.demo_out, "index.html"), 'w')
    html_txt = """"""
    ## For each domain, compute composed features and evaluate.
    for query in query_file.split(","):        
        logging.info("retrieve image of {}".format(query))
        transform = _transform(model.visual.input_resolution)
        query_img = prepare_img(query, transform)
        query_img = torch.unsqueeze(query_img, 0)    
        query_img = query_img.cuda(args.gpu, non_blocking=True)
        img_feature = m.encode_image(query_img) 
        query_img_feature = img2text(img_feature)
        composed_feature = m.encode_text_img_vis(text, query_img_feature, split_ind=id_split)
        composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
        img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
        text_feature = m.encode_text(text)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        similarity = composed_feature @ all_image_features.T
        _, indices = torch.sort(similarity, descending=True)        
        logging.info("Composed feature result")
        for i, caption in enumerate(prompt):
            logging.info("for prompt {}".format(caption))
            for j, ind in enumerate(indices[i][:8]):
                logging.info("top {} filename {}".format(j, all_image_filenames[ind]))
        image_paths = [[all_image_filenames[ind] for j, ind in enumerate(indices[i][:8])] 
                        for i, caption in enumerate(prompt)]
        html_txt += make_html(prompt, query, image_paths, args.demo_out)
    f.write(html_txt)

def make_html(prompts, query_image, images, path_html):
    import shutil
    html_all = """"""        
    for i in range(len(prompts)):
        prompt = prompts[i]            
        query_image_local = os.path.join(path_html, "images", query_image.split("/")[-1])
        query_image_local_path = os.path.join("images", query_image.split("/")[-1])
        shutil.copy(query_image, query_image_local)
        image_list = images[i]        
        html = """<table><tr>"""    
        html += """<td><p style="display:inline-block;vertical-align;font-size:20px">%s</p></td>"""%(prompt)
        html += """<td><p style="margin-right: 50px;"><img src="%s" height="100"></p></td>"""%(query_image_local_path)
        for image in image_list:
            image_local = os.path.join(path_html, "images", image.split("/")[-1])
            image_path = os.path.join("images", image.split("/")[-1])
            shutil.copy(image, image_local)
            html += """<td><img src="%s" height=%s></td>"""%(image_path, 200)
        html += """</tr></table>"""
        html_all += html
    return html_all
    #f.write(html_all)


"""
Method to return the cropped image of each object in the scene, based on the bounding boxes
- input: image_name - name of the image
- output: array containing the images of each object
"""
def cropObjectsFromImage(image_name):
    current_file_path = os.path.abspath(os.curdir)
    # logging.info(f"Current path: {current_file_path}") # on the server, it is run from the Pic2Word folder
    # previousLevel = os.path.abspath(os.path.join(current_file_path, os.pardir))
    # logging.info(f"Previous path: {previousLevel}")
    
    pathToSceneFolder = os.path.join(current_file_path, "data", "css", "scenes_bbox")
    # logging.info(f"Scene folder path: {pathToSceneFolder}")

    jsonFile = os.path.join(pathToSceneFolder, image_name.replace("png", "json"))

    with open(jsonFile, "r") as file:
        obj = json.load(file)
        bboxes = obj["bboxes"]

    pathToImageFolder = os.path.join(current_file_path, "data", "css", "images")
    # logging.info(f"Images folder path: {pathToImageFolder}")



    image = Image.open(os.path.join(pathToImageFolder, image_name))

    imgObjs = []
    for bbox in bboxes:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        cropped_image = image.crop((x1, y1, x2, y2))
        
        # resize
        newSize = (224, 224)
        cropped_image = cropped_image.resize(newSize)

        imgObjs.append(cropped_image)

    return imgObjs

"""
method to compute the image features of the batch.
Each image is taken from the images_paths and each object is cropped
    based on the boundin boxes coordinates from the scenes_bbox

- input:
    model - model used
    images_paths - array of image paths for all the images in the batch
- output: 
    batch_image_features - array of arrays of each image features of the objects in the scene (array of array of tensors - object iamge embedding)
    nr_objs - number of objects in the image
"""
def getImageFeaturesOfImage(model, imageName, preprocess_val, args):

    objImgs = cropObjectsFromImage(imageName)

    objsImgsFeatures = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for objImg in objImgs:
        objImg_preprocessed = preprocess_val(objImg)
        obj_tensor = torch.unsqueeze(objImg_preprocessed, 0).to(device)

        # Encode the cropped image
        embedding = model.encode_image(obj_tensor) # shape [(1, 768)]; type: torch.Tensor | there are 768 image features for each object
        embedding = torch.squeeze(embedding, dim=0) # convert from [[1, X]] to [X]
        objsImgsFeatures.append(embedding)

    # logging.info(f"Object image features for {imageName}: {len(objsImgsFeatures)}")

    # combine the the embeddings into a single tensor
    # image_embedding = torch.cat(objsImgsFeatures, dim=1)
    # image_embedding = torch.squeeze(image_embedding, dim=0) # convert from [[1, X]] to [X]

    # logging.info(f"image embedding: shape {image_embedding.shape}; type {type(image_embedding)}")
    # return image_embedding
    return objsImgsFeatures


def computeImageFeaturesOfBatch(model, images, images_paths, preprocess_val, args):
    image_features_list = []
    
    for image_path in images_paths:
        imageName = os.path.basename(image_path)
        image_features = getImageFeaturesOfImage(model, imageName, preprocess_val, args)

        image_features_list.append(image_features)
    # logging.info(f"Number of images in the batch: {len(images_paths)}")
    # logging.info(f"Shape of the image features list of the batch: {len(image_features_list)}")
    # logging.info(f"image features list of the batch [0]: {image_features_list[0]}; type {type(image_features_list[0])} ; shape {image_features_list[0].shape}")
    """
    for (idx, item) in enumerate(image_features_list):
        print(f"{idx}:\n\tlength = {len(item)}")
        print(f"\titem[0].length = {len(item[0])}; type: {type(item[0])}; shape: {item[0].shape}")
        if(idx == 10):
            break
        idx = idx + 1
    """

    return image_features_list

def evaluate_css(model, img2text, args, source_loader, target_loader, preprocess_val):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    all_target_paths = []
    all_answer_paths = []
    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_caption_features = []  
    all_mixture_features = []  
    all_reference_names = []
    all_captions = []     
    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean() 

    # evaluate the target data source
    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch # Target_images type: <class 'torch.Tensor'>
            
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            # logging.info(f"Target Paths: {target_paths}")
            image_features = m.encode_image(target_images)

            # logging.info(f"Image features: shape {image_features.shape}; type {type(image_features)}; device {image_features.device}; max_nr_obj: {max_nr_objs}")
            # logging.info(f"Image features [0]: shape {image_features[0].shape}; type {type(image_features[0])}")

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

    # evaluate the source data source
    with torch.no_grad():
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch
            
            for path in answer_paths:
                all_answer_paths.append(path)
            all_reference_names.extend(ref_names)
            all_captions.extend(captions)
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                target_images = target_images.cuda(args.gpu, non_blocking=True)
                target_caption = target_caption.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            # logging.info(f"Reference Names: {ref_names}")
            
            image_features = m.encode_image(target_images)
            query_image_features = m.encode_image(ref_images)
            
            # logging.info(f"Image features: shape {image_features.shape}; type {type(image_features)}; device {image_features.device}")
            # logging.info(f"Image features [0]: shape {image_features[0].shape}; type {type(image_features[0])}")
            # logging.info(f"Query Image features: shape {query_image_features.shape}; type {type(query_image_features)}; device: {query_image_features.device}; max_nr_objs: {max_nr_objs}")
            # logging.info(f"Query Image features [0]: shape {query_image_features[0].shape}; type {type(query_image_features[0])}")

            batchImageObjectsFeatures = computeImageFeaturesOfBatch(m, ref_images, answer_paths, preprocess_val, args)

            # create the text_with_blank, of format: a photo of *, and *, and * (* = nr of objects = len(batchImageObjectsFeatures[idx]))
            # and tokenize it, like in the CSS class
            for (idx, imageObjectsFeatures) in enumerate(batchImageObjectsFeatures):
                text_with_blanks = "a photo of "
                blanks = ""
                for item in imageObjectsFeatures:
                    if blanks == "": 
                        blanks += "*"
                    else: 
                        blanks += ", and *"
                text_with_blanks += blanks + f", {captions[idx]}"
                
                print(f"{idx}:\n\tlength = {len(imageObjectsFeatures)}")
                print(f"\titem[0].length = {len(imageObjectsFeatures[0])}; type: {type(imageObjectsFeatures[0])}; shape: {imageObjectsFeatures[0].shape}")
                print(f"\tText with blanks: {text_with_blanks}")
                if(idx == 10):
                    break
                idx = idx + 1


            id_split = tokenize(["*"])[0][1]

            caption_features = m.encode_text(target_caption)
            # logging.info(f"Target Caption type: {type(target_caption)}; shape: {target_caption.shape}; size: {target_caption.size()}; device {target_caption.device}")
            # logging.info(f"Caption features type: {type(caption_features)}; shape: {caption_features.shape}; size: {caption_features.size()}; device {caption_features.device}")
            # logging.info(f"Target Caption type: {target_caption}")

            query_image_tokens = img2text(query_image_features)  
            
            # logging.info(f"Query Image features: shape {query_image_features.shape}; type {type(query_image_features)}; device: {query_image_features.device}; max_nr_objs: {max_nr_objs}")
            # logging.info(f"Query Image tokens (img2text) type: {type(query_image_tokens)}; shape: {query_image_tokens.shape}; size: {query_image_tokens.size()}; device {query_image_tokens.device}")
            

            composed_feature = m.encode_text_img_retrieval(target_caption, query_image_tokens, split_ind=id_split, repeat=False)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)

            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)                         

        assert len(all_reference_names) == len(all_captions)

        logging.info("Creating the retrieval object")

        for index in range(len(all_reference_names)):
            obj = {}
            obj["query_image"] = all_reference_names[index]
            obj["query_caption"] = all_captions[index]
            obj["target_image"] = all_answer_paths[index] # might have to switch to answer path
            obj["retrieved"] = []
            retrieved_items_json_arr.append(obj)

        logging.info("Finished creating the retrieval object. Now calculating metrics")
        
        """
        logging.info(f"All image features: len {len(all_image_features)}")
        for item in all_image_features:
            logging.info(f"\t{item.shape}")

        # image_features_extended = createMatchedTensorMatrix(all_image_features)
        # logging.info(f"Extended image features: shape {image_features_extended.shape}")
        """

        metric_func = partial(get_metrics_css, 
                              image_features=torch.cat(all_image_features),
                              target_names=all_target_paths, answer_names=all_answer_paths, 
                              all_reference_names=all_reference_names, all_captions=all_captions)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        
        logging.info("Finished calculating metrics")

        for key, value in feats.items():
            metrics = metric_func(ref_features=value, feature=key)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

        logging.info("Finished everything!")
        
        # write JSON array to file
        output_file_name = "top_5_retrieved_css_images.json"
        with open(output_file_name, "w") as output_file:
            json.dump(retrieved_items_json_arr, output_file)
        
    return metrics



def get_metrics_css(image_features, ref_features, target_names, answer_names, all_reference_names, all_captions, feature):
    metrics = {}
    distances = 1 - ref_features @ image_features.T    
    # torch.cuda.empty_cache()
    # logging.info(f"Metrics - Before Argsort. Distance variable device: {distances.device}")
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    # logging.info("Metrics - after Argsort")
    # logging.info(f'Sorted_indexes length: {sorted_indices.shape}')
    # logging.info(f"sorted_index_names[0]: {np.array(target_names)[0]}")
    # logging.info(f'Target_names length: {len(target_names)}')
    sorted_index_names = np.array(target_names)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(answer_names), len(target_names)).reshape(len(answer_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(answer_names)).int())

    assert len(all_reference_names) == len(all_captions) and feature in ["composed", "text", "image", "mixture"]
    
    N = 5
    for index in range(len(all_reference_names)):
        obj = {}
        obj[feature] = sorted_index_names[index][0:N].tolist() # to convert tensor to np array: cpu().detach().numpy()
        retrieved_items_json_arr[index]["retrieved"].append(obj)
        

    # Compute the metrics
    for k in [1, 5, 10, 50, 100]:
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    return metrics

