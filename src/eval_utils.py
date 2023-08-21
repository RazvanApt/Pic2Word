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


def evaluate_imgnet_retrieval(model, img2text, args, prompt, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    all_image_features = []  
    all_target_labels = []      
    m = model.module if args.distributed or args.dp else model
    n_class = 1000
   
    with torch.no_grad():
        ## Extract target image features. 
        for batch in tqdm(target_loader):
            images, labels = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            all_target_labels.append(labels)
            logit_scale = m.logit_scale.exp()
            logit_scale = logit_scale.mean()   

        ## Extract query features 
        for p_ind, p in enumerate(prompt):            
            ## which token has to be replaced with image features
            id_split = tokenize(["*"])[0][1]
            text = tokenize(p).view(1, -1)
            text = text.cuda(args.gpu, non_blocking=True)
            ## text only features (domain name only)
            text_only = p.replace("*", "")
            text_only = tokenize(text_only).view(1, -1)            
            text_only = text_only.cuda(args.gpu, non_blocking=True)                        
            text_only_features = m.encode_text(text_only)
            text_only_features_normed = text_only_features / text_only_features.norm(dim=-1, keepdim=True)

            all_query_features = []
            all_query_image_features = []
            all_query_mixture_features = []
            all_query_labels = []
            all_text_features = []
            for batch in tqdm(query_loader):
                images, labels = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    labels = labels.cuda(args.gpu, non_blocking=True)
                ## Label is decided by class label and images' domain
                labels += n_class * p_ind
                image_features = m.encode_image(images)
                 ## Composed feature extraction
                image_features_query = img2text(image_features)                      
                composed_feature = m.encode_text_img_retrieval(text, image_features_query, split_ind=id_split)                            
                composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
                ## Image feature only
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
                ## average of image and text features
                mixture_features = image_features + text_only_features_normed
                mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)       

                all_text_features.append(text_only_features_normed.repeat((image_features.shape[0], 1)))
                all_query_features.append(composed_feature)
                all_query_image_features.append(image_features)
                all_query_mixture_features.append(mixture_features)
                all_query_labels.append(labels)

            metric_func = partial(get_metrics_imgnet, 
                image_features=torch.cat(all_image_features), 
                query_labels=torch.cat(all_query_labels),
                target_labels=torch.cat(all_target_labels),
                )

            feats = {'composed': torch.cat(all_query_features), 
                    'image': torch.cat(all_query_image_features),
                    'text': torch.cat(all_text_features),
                    'mixture': torch.cat(all_query_mixture_features)}        

            for key, value in feats.items():
                metrics = metric_func(query_features=value)
                logging.info(
                f"Eval {key} Feature"
                + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_coco(model, img2text, args, loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_mixture_features = []  
    all_composed_features_with_class = []  
    all_text_full_features = [] 

    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()
    with torch.no_grad():
        for batch in tqdm(loader):
            images, region_images, text_full, text_with_blank, text_with_blank_query, filename, raw_text = batch            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                region_images = region_images.cuda(args.gpu, non_blocking=True)
                text_full = text_full.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                text_with_blank_query = text_with_blank_query.cuda(args.gpu, non_blocking=True)

            ## Target image features 
            image_features = m.encode_image(images)             
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
            id_split = tokenize(["*"])[0][1]
            ## Composed image features
            query_image_features = m.encode_image(region_images)
            query_image_tokens = img2text(query_image_features)          
            composed_feature_with_class = m.encode_text_img_retrieval(text_with_blank_query, query_image_tokens, split_ind=id_split, repeat=False)                        
            composed_feature_with_class = composed_feature_with_class / composed_feature_with_class.norm(dim=-1, keepdim=True)        
            ## Text only features
            text_full_features = m.encode_text(text_full)
            text_full_features = text_full_features / text_full_features.norm(dim=-1, keepdim=True)            
            ## Query only features
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)                               
            ## Mixed featurs
            mixture_features = query_image_features + text_full_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)            

            all_image_features.append(image_features.cpu())
            all_text_full_features.append(text_full_features.cpu())       
            all_query_image_features.append(query_image_features.cpu())
            all_mixture_features.append(mixture_features.cpu())                        
            all_composed_features_with_class.append(composed_feature_with_class.cpu())            

        metric_func = partial(get_metrics_coco, 
                image_features=torch.cat(all_image_features), 
                logit_scale=logit_scale
                )
        feats = {'composed': torch.cat(all_composed_features_with_class), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_text_full_features),
                 'mixture': torch.cat(all_mixture_features)}        

        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_cirr(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_raw_captions = []
    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, answer_paths, raw_captions = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for path in ref_paths:
                all_ref_paths.append(path)
            for path in answer_paths:
                all_answer_paths.append(path)
            for cap in raw_captions:
                all_raw_captions.append(cap)

            caption_features = m.encode_text(caption_only)
            ## Composed features
            query_image_features = m.encode_image(ref_images)
            query_image_tokens = img2text(query_image_features)
            composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)                

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            mixture_features = query_image_features + caption_features            
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)                        

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        
        metric_func = partial(get_metrics_cirr, 
                image_features=torch.cat(all_image_features), 
                reference_names=all_ref_paths, 
                index_names=all_target_paths, 
                target_names=all_answer_paths)

        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics


def evaluate_cirr_test(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_composed_plus_image_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_ids = []

    m = model.module if args.distributed or args.dp else model   
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, pairids = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for ids in pairids:
                all_ids.append(ids)
            for path in ref_paths:
                all_ref_paths.append(path)

            caption_features = m.encode_text(caption_only)
            query_image_features = m.encode_image(ref_images)

            if args.eval_combiner:
                composed_feature = img2text(query_image_features, caption_features)
            else:
                query_image_tokens = img2text(query_image_features)
                composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)            

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        res_all = {}
        metrics_func = partial(get_cirr_testoutput, 
                               image_features=torch.cat(all_image_features),
                               reference_names=all_ref_paths,
                               index_names=all_target_paths,
                               id_names=all_ids)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}        
        for key, value in feats:
            res_all[key] = metrics_func(ref_features=value)
    return res_all


def evaluate_fashion(model, img2text, args, source_loader, target_loader):
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

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

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
            image_features = m.encode_image(target_images)
            query_image_features = m.encode_image(ref_images)
            id_split = tokenize(["*"])[0][1]

            caption_features = m.encode_text(target_caption)                            
            query_image_tokens = img2text(query_image_features)          
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

        for index in range(len(all_reference_names)):
            obj = {}
            obj["query_image"] = all_reference_names[index]
            obj["query_caption"] = all_captions[index]
            obj["target_image"] = all_answer_paths[index] # might have to switch to answer path
            obj["retrieved"] = []
            retrieved_items_json_arr.append(obj)

        metric_func = partial(get_metrics_fashion, 
                              image_features=torch.cat(all_image_features),
                              target_names=all_target_paths, answer_names=all_answer_paths, 
                              all_reference_names=all_reference_names, all_captions=all_captions)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        


        for key, value in feats.items():
            metrics = metric_func(ref_features=value, feature=key)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


        
        # write JSON array to file
        output_file_name = "top_5_retrieved_images.json"
        with open(output_file_name, "w") as output_file:
            json.dump(retrieved_items_json_arr, output_file)
        
    return metrics

"""
TODO: create the evaluation method for CSS
get the json file that contains the bounding boxes and crop each object from the image accordingly 
get the image features of each cropped image
create a tensor that contains all individual image features 
feed this tensor to the AI model
"""


def createMatchedTensorMatrix(list):
    # Convert tensors to lists
    arr_as_lists = [t.tolist() for t in list]

    # Determine the maximum row size
    max_row_size = max(len(row) for row in arr_as_lists)

    # Pad the rows with zeros to make them the same size
    padded_arr = [row + [0] * (max_row_size - len(row)) for row in arr_as_lists]

    # Convert the padded list to a tensor
    return torch.tensor(padded_arr)

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
    batch_image_features - array of arrays of each image features of the objects in the scene
    nr_objs - number of objects in the image
"""

# idea: create tensor with everything and then call the encode

def getImageFeaturesOfImage(model, imageName, preprocess_val, args):

    objImgs = cropObjectsFromImage(imageName)
    nr_objs = len(objImgs)

    objsImgsFeatures = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for objImg in objImgs:
        objImg_preprocessed = preprocess_val(objImg)
        obj_tensor = torch.unsqueeze(objImg_preprocessed, 0).to(device)

        # Encode the cropped image
        embedding = model.encode_image(obj_tensor) # shape [(1, 768)]; type: torch.Tensor | there are 768 image features for each object
        objsImgsFeatures.append(embedding)

    # logging.info(f"Object image features for {imageName}: {len(objsImgsFeatures)}")

    # combine the the embeddings into a single tensor
    image_embedding = torch.cat(objsImgsFeatures, dim=1)
    image_embedding = torch.squeeze(image_embedding, dim=0) # convert from [[1, X]] to [X]

    # logging.info(f"image embedding: shape {image_embedding.shape}; type {type(image_embedding)}")
    return image_embedding, nr_objs



def computeImageFeaturesOfBatch(model, images, images_paths, preprocess_val, args):
    batch_image_features = []
    image_features_list = []
    max_nr_objs = 0 # the maximum nr of objects in an image of the batch
    for image_path in images_paths:
        imageName = os.path.basename(image_path)
        image_features, nr_objs = getImageFeaturesOfImage(model, imageName, preprocess_val, args) # This is a torch.Tensor
        # size of image features: [768 x NR_OBJS_IN_IMG]
        
        max_nr_objs = nr_objs if max_nr_objs < nr_objs else max_nr_objs

        image_features_list.append(image_features)
    logging.info(f"Number of images in the batch: {len(images_paths)}")
    # logging.info(f"Shape of the image features list of the batch: {len(image_features_list)}")
    # logging.info(f"image features list of the batch [0]: {image_features_list[0]}; type {type(image_features_list[0])} ; shape {image_features_list[0].shape}")
    
    """
    get the maximum row length of the image_features_list (which is a matrix)
    use padding for the rest of the rows to get to max
    use torch.cat on that and check to have size = [nr_of images, .....]
    """
    # size of batch_image_features shold be [NR_IMAGES, 768 x NR_OBJECTES_PER_IMAGE]

    batch_image_features = createMatchedTensorMatrix(image_features_list)

    # batch_image_features = torch.cat((image_features_list), dim=0)
    # batch_image_features = torch.stack((image_features_list))


    logging.info(f"Batch shape: {batch_image_features.shape}; and batch type: {type(batch_image_features)}")

    """
    # Downsample using max pooling
    max_pooled_tensor = F.adaptive_max_pool1d(original_tensor.unsqueeze(0), 768).squeeze(0)

    # Downsample using average pooling
    avg_pooled_tensor = F.adaptive_avg_pool1d(original_tensor.unsqueeze(0), 768).squeeze(0)

    logging.info(f"Batch shape: {batch_image_features.shape}; and batch type: {type(batch_image_features)}")
    """
    return batch_image_features, max_nr_objs

def computeImageFeaturesOfBatch_v1(model, images, images_paths, preprocess_val, args):
    batch_image_features = []
    DIMENSION = 1 # for torch.cat method
    
    for image_path in images_paths:
        """
        for each image in the batch, get the objects
        for each object, compute the image features, and combine the 
            image features into one array for every image
        """
        imageName = os.path.basename(image_path)
        objImgs = cropObjectsFromImage(imageName)
        objsImgsFeatures = []

        for objImg in objImgs:
            objImgEncoded = model.encode_image(torch.unsqueeze(preprocess_val(objImg).cuda(args.gpu, non_blocking=True), 0))
            objsImgsFeatures = torch.cat((torch.tensor(objsImgsFeatures).cuda(args.gpu, non_blocking=True), objImgEncoded), dim=DIMENSION)


        # combine the features of every object in the batch into one array
        batch_image_features = torch.cat((torch.tensor(batch_image_features).cuda(args.gpu, non_blocking=True), torch.tensor(objsImgsFeatures).cuda(args.gpu, non_blocking=True)), dim=DIMENSION)

        # logging.info(f"Batch image features shape: {batch_image_features.shape}")

    # TODO: downsample to match the nr of columns required by the IMG2TEXT model (768)

    return batch_image_features


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
            # image_features = m.encode_image(target_images)
            image_features, max_nr_objs = computeImageFeaturesOfBatch(m, target_images, target_paths, preprocess_val, args)
            image_features = image_features.cuda()
            logging.info(f"Image features: shape {image_features.shape}; type {type(image_features)}; device {image_features.device}; max_nr_obj: {max_nr_objs}")
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
            
            # image_features = m.encode_image(target_images)
            # query_image_features = m.encode_image(ref_images)
            image_features, _ = computeImageFeaturesOfBatch(m, ref_images, answer_paths, preprocess_val, args)
            image_features = image_features.cuda()
            query_image_features, max_nr_objs = computeImageFeaturesOfBatch(m, ref_images, ref_names, preprocess_val, args)
            # query_image_features = query_image_features.cuda()

            logging.info(f"Image features: shape {image_features.shape}; type {type(image_features)}; device {image_features.device}")
            # logging.info(f"Image features [0]: shape {image_features[0].shape}; type {type(image_features[0])}")
            # logging.info(f"Query Image features: shape {query_image_features.shape}; type {type(query_image_features)}; device: {query_image_features.device}; max_nr_objs: {max_nr_objs}")
            # logging.info(f"Query Image features [0]: shape {query_image_features[0].shape}; type {type(query_image_features[0])}")

            id_split = tokenize(["*"])[0][1]

            caption_features = m.encode_text(target_caption)
            logging.info(f"Target Caption type: {type(target_caption)}; shape: {target_caption.shape}; size: {target_caption.size()}; device {target_caption.device}")
            logging.info(f"Caption features type: {type(caption_features)}; shape: {caption_features.shape}; size: {caption_features.size()}; device {caption_features.device}")


            # query_image_tokens = img2text(query_image_features)  
            dynamicIMG2TEXT = DynamicIM2TEXT(max_nr_objs)
            dynamicIMG2TEXT.eval()

            query_image_tokens = dynamicIMG2TEXT(query_image_features)
            query_image_tokens = query_image_tokens.cuda()
            query_image_features = query_image_features.cuda()
            logging.info(f"Query Image features: shape {query_image_features.shape}; type {type(query_image_features)}; device: {query_image_features.device}; max_nr_objs: {max_nr_objs}")
            logging.info(f"Query Image tokens (img2text) type: {type(query_image_tokens)}; shape: {query_image_tokens.shape}; size: {query_image_tokens.size()}; device {query_image_tokens.device}")
            
            
            # TODO: upsample caption features to match the size of query image features
            
            caption_features_expanded = caption_features.repeat(1, max_nr_objs)
            logging.info(f"Caption geatures extended: shape {caption_features_expanded.shape}; device {caption_features_expanded.device}")

            composed_feature = m.encode_text_img_retrieval(target_caption, query_image_tokens, split_ind=id_split, repeat=False)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            
            # caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
            caption_features_expanded = caption_features_expanded / caption_features_expanded.norm(dim=-1, keepdim=True)

            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            mixture_features = query_image_features + caption_features_expanded
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            all_caption_features.append(caption_features_expanded)
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

        logging.info("Finished computing features. Now calculating metrics")
        
        logging.info(f"All image features: len {len(all_image_features)}")
        for item in all_image_features:
            logging.info(f"\t{item.shape}")

        image_features_extended = createMatchedTensorMatrix(all_image_features)
        logging.info(f"Extended image features: shape {image_features_extended.shape}")

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


def get_metrics_coco(image_features, ref_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale.cpu() * image_features @ ref_features.t()).detach().cpu()
    logits_per_ref = logits_per_image.t().detach().cpu()
    logits = {"image_to_ref": logits_per_image, "ref_to_image": logits_per_ref}
    ground_truth = torch.arange(len(ref_features)).view(-1, 1)
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10, 50, 100]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics


def get_metrics_fashion(image_features, ref_features, target_names, answer_names, all_reference_names, all_captions, feature):
    metrics = {}
    distances = 1 - ref_features @ image_features.T    
    logging.info(f"Metrics - Before Argsort. Distance variable device: {distances.device}")
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    logging.info("Metrics - after Argsort")
    logging.info(f'Sorted_indexes length: {sorted_indices.shape}')
    logging.info(f"sorted_index_names[0]: {np.array(target_names)[0]}")
    logging.info(f'Target_names length: {len(target_names)}')
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


def get_metrics_cirr(image_features, ref_features, reference_names, index_names, target_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), 
        len(index_names)).reshape(len(target_names), -1))        
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), 
        len(index_names) - 1).reshape(len(target_names), -1))

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    for k in [1, 5, 10, 50, 100]:
        metrics[f"recall_R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100

    return metrics


def get_cirr_testoutput(image_features, ref_features, reference_names, index_names, id_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    result_dict = {"version": "rc2", "metric": "recall"}
    for ind in range(len(id_names)):
        pairid = str(id_names[ind].item())
        result_dict[pairid] = []
        for t in range(50):
            result_dict[pairid].append(sorted_index_names[ind][t].replace(".png", ""))
    return result_dict


def get_metrics_imgnet(query_features, image_features, query_labels, target_labels):
    metrics = {}
    num_classes = 7000
    query_onehot = F.one_hot(query_labels, num_classes=num_classes).float()
    target_onehot = F.one_hot(target_labels, num_classes=num_classes).float()
    batches = [(query_features[x:x+100], query_onehot[x:x+100]) for x in range(0, len(query_features), 100)]
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] = 0
        metrics[f"Real2Sketch_P@{k}"] = 0
    for batch in batches:
        feats, labels = batch[0], batch[1]
        logits_per_query = (feats @ image_features.t()).detach().cpu()
        label_matrix = (labels @ target_onehot.t()).detach().cpu()                
        ranking = torch.argsort(logits_per_query, descending=True)
        for k in [1, 5, 10, 50, 100, 200]:
            matrix_k = torch.zeros_like(label_matrix)
            rank_k = ranking[:, :k]
            matrix_k[torch.arange(matrix_k.size(0)).unsqueeze(1), rank_k] = 1
            consistency = matrix_k * label_matrix
            num_correct = torch.sum(consistency, dim=1)
            num_predicted = torch.sum(matrix_k, dim=1)            
            num_total = torch.sum(label_matrix, dim=1)
            recall = torch.mean(num_correct / (num_total+1e-5))
            precision = torch.mean(num_correct / num_predicted)
            metrics[f"Real2Sketch_R@{k}"] += recall * len(feats)
            metrics[f"Real2Sketch_P@{k}"] += precision * len(feats)
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] /= len(query_features)
        metrics[f"Real2Sketch_P@{k}"] /= len(query_features)
    return metrics


def get_metrics_css(image_features, ref_features, target_names, answer_names, all_reference_names, all_captions, feature):
    metrics = {}
    distances = 1 - ref_features @ image_features.T    
    # torch.cuda.empty_cache()
    logging.info(f"Metrics - Before Argsort. Distance variable device: {distances.device}")
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    logging.info("Metrics - after Argsort")
    logging.info(f'Sorted_indexes length: {sorted_indices.shape}')
    logging.info(f"sorted_index_names[0]: {np.array(target_names)[0]}")
    logging.info(f'Target_names length: {len(target_names)}')
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

