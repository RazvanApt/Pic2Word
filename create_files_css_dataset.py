"""

get the cap.{split}.json files from css/json (split = ['train', 'val', 'test'])
combine the files split.{split}.json into one file

name the new files for 'dress' type of cloths
TODO: omit the cap.test.file because it doesn't have the 'target' key in the structure of the items

"""
import json
import os

css_path = "D:\Razvan\Documents\CURSURI UAB\Dissertation\Pic2Word\data\css"
is_path = os.path.join(css_path, "image_splits")
json_path = os.path.join(css_path, "json")

output_is_filename = "new_split.dress.val.json"
output_j_filename = "new_cap.dress.val.json"


total_is_arr = []
total_j_arr = []

for split in ['train', 'val']:
    with open(json_path + '\\' + f"cap.{split}.json", "r") as caption_file:
        temp_arr = json.load(caption_file)
    total_j_arr += temp_arr

with open(json_path + '\\' + output_j_filename, "w") as new_caption_file:
    json.dump(total_j_arr, new_caption_file)

used_images = []
for item in total_j_arr:
    if ('target' in item and item['target'] not in used_images):
        used_images.append(item['target'])

    if ('candidate' in item and item['candidate'] not in used_images):
        used_images.append(item['candidate'])

with open (is_path + '\\' + output_is_filename, "w") as new_image_split_file:
    json.dump(used_images, new_image_split_file)


print("DONE")


