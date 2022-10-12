

import os
import shutil
import csv
import requests
import io
import json
import PIL.Image


PATH_TO_METADATA = os.path.abspath("./metadata")
PATH_TO_DIR = os.path.abspath(".")

TRAIN_SAMPLES = 2000    ##out of 3,318,333 potential
VALID_SAMPLES = 800     ##out of 15,840 potential
IMAGE_SIZE = (256, 256)


def get_images_and_save(split:str, n_samples:int, 
                        image_size:tuple, interpolation=PIL.Image.Resampling.BICUBIC,
                        fresh_dir:bool=True):
    "Retrieves images from urls and saves them to a directory. Saves the relative image path, url, and caption for all successful downloads."
    OUT_DIR = os.path.join(PATH_TO_DIR, f"{split}_images/")
    log_file = os.path.join(PATH_TO_METADATA, f"{split}-import-log.txt")

    # empty log file if already exists
    open(log_file, 'w').close()

    # load in the downloaded 'metadata' - ie, url - caption pairs
    meta = _import_metadata(split, n_samples)
    urls = [u for l, u in meta]
    labels = [l for l, u in meta]

    # if out_dir already exits, remove for a clean slate
    if fresh_dir:  
        if os.path.isdir(OUT_DIR):
            shutil.rmtree(OUT_DIR) # remove
        os.mkdir(OUT_DIR) # make a new empty dir
    else:
        if not os.path.isdir(OUT_DIR):
            os.mkdir(OUT_DIR)

    saved_paths, saved_labels, saved_urls = [], [], []
    total = 0
    for ix, url in enumerate(urls):
        # before saving image to disk, determine the image name & path
        if total < 10:
            str_total = "00"+str(total)
        elif 9 < total < 100:
            str_total = "0"+str(total)
        elif total > 99:
            str_total = str(total) ##remove this stuff
        img_path = os.path.join(OUT_DIR+str(total)+".jpg")
        
        try: #try the url, if successful open the image and save to the img_path
            res = requests.get(url, timeout=60)
            img = PIL.Image.open(io.BytesIO(res.content))
            if image_size is not None:
                img = img.resize((image_size), resample=interpolation)
            img.save(img_path)

            # save the image path and label and update the counter
            saved_paths.append(img_path.strip('.'))
            saved_labels.append(labels[ix])
            saved_urls.append(url)
            total += 1

        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"index: {ix}; url: {url}; error: {e}\n")

    new_meta = dict(paths = saved_paths, labels = saved_labels, urls = saved_urls)

    return new_meta

def _import_metadata(split, n_samples):
    """Import downloaded url - caption pairs from https://ai.google.com/research/ConceptualCaptions/download"""
    meta = []
    
    if split == "train":
        filepath = f"{PATH_TO_METADATA}/Train_GCC-training.tsv"
    elif split == "valid":
        filepath = f"{PATH_TO_METADATA}/Validation_GCC-1.1.0-Validation.tsv"

    with open(filepath, 'r', encoding="utf-8") as f:
        rd_file = csv.reader(f, delimiter="\t", quotechar='"')
        for line in rd_file:
            meta.append(line)
            if len(meta) == n_samples:
                break
    return meta

def save_meta(meta_dict, split):
    with open(f'{PATH_TO_METADATA}/{split}_meta.json', 'w') as f:
        json.dump(meta_dict, f, sort_keys=True, indent=4)

def load_meta(path):
    """Load in 'train' or 'valid' metadata. Metadata includes paths to images and their captions."""
    with open(path, 'r') as f:
        data = json.load(f) ##?
        return data


if __name__=='__main__':
    train_meta = get_images_and_save(split="train", 
                                n_samples=TRAIN_SAMPLES,
                                image_size=IMAGE_SIZE,
                                fresh_dir=True)


    valid_meta = get_images_and_save(split="valid",
                                    n_samples=VALID_SAMPLES,
                                    image_size=IMAGE_SIZE,
                                    fresh_dir=True)
    
    save_meta(train_meta, 'train')
    save_meta(valid_meta, 'valid')

    






