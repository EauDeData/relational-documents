import os
import re
import json
import nltk
import random
import pytesseract
import unicodedata

import torch.utils.data
from torchvision.transforms.functional import pil_to_tensor
import PIL.ImageOps as imOps
import numpy as np
from transformers import BertTokenizer

import math
from PIL import Image
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import torchvision.transforms as transforms

from data.defaults import WORD_IMAGE_SIZE, BASE_JSONS, REPLACE_PATH_EXPR, MAX_WORDS

RN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet typically expects 224x224 input size
    transforms.ToTensor(),         # Converts PIL image to PyTorch tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean values for normalization (ImageNet)
        std=[0.229, 0.224, 0.225]    # Std values for normalization (ImageNet)
    )
])

WORD_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(WORD_IMAGE_SIZE),
    transforms.ToTensor(),         # Converts PIL image to PyTorch tensor
])


# Make sure to download stopwords and punkt if you haven't already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
random.seed(42)

SPANISH_STOPWORDS = set(stopwords.words('spanish'))
MIN_WORD_LENGTH = 2


def crop_image_to_patches(image: Image.Image, num_patches: int):
    """
    Crops a PIL image into N approximately equal patches.

    Args:
        image (Image.Image): The input image to crop.
        num_patches (int): The number of patches to divide the image into.

    Returns:
        list: A list of cropped image patches as PIL Images.
    """

    # Calculate number of rows and columns
    rows = math.isqrt(num_patches)
    cols = math.ceil(num_patches / rows)

    if rows * cols < num_patches:
        rows += 1

    # Calculate patch size
    img_width, img_height = image.size
    patch_width = img_width // cols
    patch_height = img_height // rows

    patches = []
    for row in range(rows):
        for col in range(cols):
            # Calculate the coordinates for each patch
            left = col * patch_width
            upper = row * patch_height
            right = left + patch_width
            lower = upper + patch_height

            # Make sure the last patches reach the edges of the image
            if col == cols - 1:
                right = img_width
            if row == rows - 1:
                lower = img_height

            # Crop the patch and append it
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)

            # Stop if we've reached the desired number of patches
            if len(patches) == num_patches:
                break
        if len(patches) == num_patches:
            break

    return patches

def split_by_punctuation_and_spaces(sentence):
    # Split by any punctuation or spaces using regex
    tokens = re.split(r'\W+', sentence)

    # Remove empty tokens
    tokens = [token for token in tokens if token if not token in SPANISH_STOPWORDS]

    return tokens

def curate_token(token, stemmer):
    return stemmer(token)

def sanitize_string(input_string: str) -> str:
    # Normalize the input string to its decomposed form (NFD), separating characters from their accents
    normalized_string = unicodedata.normalize('NFD', input_string)

    # Filter out all diacritical marks (i.e., non-spacing marks like accents)
    sanitized_string = ''.join(char for char in normalized_string if not unicodedata.combining(char))

    return sanitized_string

class BOEDataset(torch.utils.data.Dataset):
    def __init__(self, json_split_txt, base_jsons=BASE_JSONS,
                 replace_path_expression=REPLACE_PATH_EXPR):
        self.replace = eval(replace_path_expression)
        self.documents = []
        for line in tqdm(open(os.path.join(base_jsons, json_split_txt)).readlines()):
            path = os.path.join(base_jsons, line.strip()).replace('jsons_gt', 'graphs_gt')
            # if len(self.documents) > 65: break
            # Avoid empty files
            if len(json.load(open(path))['pages']['0']) > 1:
                self.documents.append(path)


        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def __len__(self):
        return len(self.documents)


    def tokenize_sentence(self, sentence):
        encoded_tokens = self.tokenizer.encode(' '.join(sentence), add_special_tokens=True)
        return encoded_tokens

    def crop_dataset(self):

        for idx in tqdm(list(range(len(self))), desc='Cropping dataset'):
            path = self.documents[idx]

            json_data = json.load(open(path))
            impath = (json_data['path'].replace(*self.replace)
                      .replace('images', 'numpy').replace('.pdf', '.npz'))

            folder_path = impath.replace('.npz', '_cropped_ocrs').replace('numpy', 'crops')
            os.makedirs(folder_path, exist_ok=True)
            crop_path = os.path.join(folder_path, 'crop.png')

            if not os.path.exists(crop_path):
                page = json_data['topic_gt']["page"]
                segment = json_data['topic_gt']["idx_segment"]
                x, y, x2, y2 = json_data['pages'][page][segment]['bbox']

                image = Image.fromarray(np.load(impath)[page][y:y2, x:x2])
                image.save(crop_path)
            else:
                image = Image.open(crop_path)

            df = pytesseract.image_to_data(image, lang='spa_old', output_type=pytesseract.Output.DATAFRAME)
            df = df[df['level'] == 5].reset_index()

            words_path = os.path.join(folder_path, 'words')
            metadata_path = os.path.join(folder_path, 'metadata')

            os.makedirs(words_path, exist_ok=True)
            os.makedirs(metadata_path, exist_ok=True)

            for i in df.index:

                left = df.at[i, 'left']
                top = df.at[i, 'top']
                width = df.at[i, 'width']
                height = df.at[i, 'height']
                right = left + width
                bottom = top + height
                # Crop the image using the bounding box coordinates
                try:
                    cropped_image = image.crop((left, top, right, bottom))
                    cropped_image.save(os.path.join(words_path, f"{i}.png"))
                    with open(os.path.join(metadata_path, f'{i}.txt'), 'w') as handler:
                        handler.write('\n'.join([f'left: {left}', f'top: {top}',
                                                 f'width: {left}', f'height: {left}',
                                                 f"word: {df.at[i, 'text']}"]))
                except SystemError:
                    continue

    def __getitem__(self, idx):
        path = self.documents[idx]
        json_data = json.load(open(path))

        # Assumes crop already exists
        impath = (json_data['path'].replace(*self.replace)
                  .replace('images', 'numpy').replace('.pdf', '.npz'))
        folder_path = impath.replace('.npz', '_cropped_ocrs').replace('numpy', 'crops')
        crop_path = os.path.join(folder_path, 'crop.png')
        image = Image.open(crop_path)

        wordcrops = crop_image_to_patches(image, MAX_WORDS)

        # crop_path = ((json_data['path'].replace(*self.replace)
        #               .replace('images', 'numpy')
        #               .replace('.pdf', '.npz'))
        #              .replace('.npz', '_cropped_ocrs')
        #              .replace('numpy', 'crops')
        #              )
        # words = os.path.join(crop_path, 'words')
        # # HGHAHAHAHHHAHAHAHAHA TRY TO READ THIS CODE FUCKER
        # croplist = [os.path.join(crop_path, words, x) for x in
        #             sorted(os.listdir(words), key=lambda x: int(x.split('.')[0]))]
        # metadatas = [imname.replace('words', 'metadata').replace('.png', '.txt')
        #              for imname in croplist]
        #
        # metadata_files = [open(metadata).readlines() for metadata in metadatas]
        #
        # # Assuming 'croplist' corresponds to images, and 'words_thing' has matching entries
        # wordcrops = [imOps.invert(Image.open(imname).convert('RGB').resize(WORD_IMAGE_SIZE))
        #              for imname, metadata in zip(croplist, metadata_files)
        #              if len(metadata[-1].strip().split(': ')[-1]) > MIN_WORD_LENGTH]
        # # random.shuffle(wordcrops) # WARNING: WE HAVE A SEQUENCE RANDOMIZER HERE!!
        # wordcrops = wordcrops[:MAX_WORDS]
        # padding = [Image.new('RGB', (5, 5)).resize(WORD_IMAGE_SIZE)] * (MAX_WORDS - len(wordcrops))
        # wordcrops = wordcrops + padding

        query = json_data['query']
        query_str = [q for q in split_by_punctuation_and_spaces(query)]

        return {'image': RN_TRANSFORMS(image), 'wordcrops': [RN_TRANSFORMS(wc).squeeze() for wc in wordcrops], 'query': self.tokenize_sentence(query_str)}
               # {'pil_image': image, 'pil_wordcrops': wordcrops, 'plain_query_list': ['[CLS]'] + query_str})

if __name__ == '__main__':
    data = BOEDataset('test.txt')[10]
    print(data)
