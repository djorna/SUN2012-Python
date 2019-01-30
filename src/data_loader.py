import tarfile
from tqdm import tqdm
import numpy as np
import cv2
import glob
from collections import OrderedDict
import json
import xml.etree.ElementTree as ET
import pandas as pd
import os

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numpy.random import random, randint



class SUNLoader:

  def __init__(self, file_csv, groups_json, root_dir=".", np_dir="data/numpy"):

    self.known_cats = OrderedDict(json.load(open(groups_json))["groups"])
    self.n_categories = len(self.known_cats) + 2 # known categories + other + unlabelled

    self.cat_index = {key: num + 2 for num, key in enumerate(self.known_cats)}
    self.cat_index["other"] = 1
    self.cat_index["unlabelled"] = 0

    self.known_cats_rev = {}
    for cat, group in self.known_cats.items():
      for item in group:
        self.known_cats_rev[item] = cat
      
    self.file_df = pd.read_csv(file_csv, index_col=0)
    self.n_images = len(self.file_df)
    self.root_dir = root_dir
    self.np_dir = np_dir


  def load_SUN2012(self, filepath=None, destination=None):
    """
    Load SUN2012 dataset from tar file, extracting into .npy files
    """

    print("Extracting tar files...", end="")
    with tarfile.open(filepath) as tar:
      for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
        tar.extract(member, destination)
    print("done.")


  def to_numpy(self, root_dir=None, np_dir=None, limit=None):
    root_dir = self.root_dir if root_dir is None else root_dir
    np_dir = self.np_dir if np_dir is None else np_dir

    image_dir = np_dir + "/images/"
    label_dir = np_dir + "/labels/"
    for folder in [np_dir, image_dir, label_dir]:
      if not os.path.exists(folder):
        os.makedirs(folder)

    for index, row in tqdm(self.file_df.iterrows()):
      file_id = row.name
      image_path = os.path.join(root_dir, row["image_path"])
      label_path = os.path.join(root_dir, row["label_path"])
      shape = eval(row["shape"])
      img = load_image(image_path)
      label = self.get_mask(label_path, shape)
      image_path_np = image_dir + file_id + ".npy"
      label_path_np = label_dir + file_id + ".npy"
      np.save(image_path_np, img)
      np.save(label_path_np, label)

      if limit is not None:
        if index >= limit:
          break

    print("Saved images in {} and labels in {}.".format(image_dir, label_dir))


  def to_numpy_select(self, ids, root_dir=None, np_dir=None):
    root_dir = self.root_dir if root_dir is None else root_dir
    np_dir = self.np_dir if np_dir is None else np_dir
    image_dir = np_dir + "/images/"
    label_dir = np_dir + "/labels/"
    for folder in [np_dir, image_dir, label_dir]:
      if not os.path.exists(folder):
        os.makedirs(folder)

    select_rows = self.file_df.query("id in @ids")

    for index, row in tqdm(select_rows.iterrows()):
      file_id = row.name
      image_path = os.path.join(root_dir, row["image_path"])
      label_path = os.path.join(root_dir, row["label_path"])
      shape = eval(row["shape"])
      img = load_image(image_path)
      label = self.get_mask(label_path, shape)
      image_path_np = np_dir + "/images/" + file_id + ".npy"
      label_path_np = np_dir + "/labels/" + file_id + ".npy"
      np.save(image_path_np, img)
      np.save(label_path_np, label)

    print("Saved numpy files to {}".format(np_dir))


  def sort_order(self, cat):
    return self.cat_index[cat]


  def label_path(self, label_id, root_dir=None):
    root_dir = self.root_dir if root_dir is None else root_dir
    return os.path.join(root_dir, self.file_df.loc[label_id]["label_path"])


  def image_path(self, image_id, root_dir=None):
    root_dir = self.root_dir if root_dir is None else root_dir
    return os.path.join(root_dir, self.file_df.loc[label_id]["image_path"])


  def get_mask(self, filename, shape):
    tree = ET.parse(filename)
    root = tree.getroot()
    mask = np.zeros(shape, dtype=np.uint8)
    objects = root.findall("object")
    objects_cats = [(obj, self.get_category(obj.find("name").text.strip().lower()))
                    for obj in objects]
    
    for obj, category in objects_cats:    
      deleted = obj.find("deleted").text.strip()
      if deleted != "0":
        continue        
      
      index = self.sort_order(category)
      polygons = obj.findall("polygon")       

      for poly in polygons:
        pts = poly.findall("pt")
        n_pts = len(pts)
        pts_array = np.empty((1, n_pts, 2), dtype=np.int32)
        
        for i in range(n_pts):
          x = int(pts[i].find("x").text)
          y = int(pts[i].find("y").text)
          pts_array[0,i,:] = (x, y)

        # draw the polygon with OpenCV
        cv2.fillPoly(mask, pts_array, index) # set value to index
    return mask


  def get_name_list(self, file_id, root_dir=None):
    root_dir = self.root_dir if root_dir is None else root_dir
    filename = self.label_path(file_id, root_dir)
    tree = ET.parse(filename)
    root = tree.getroot()
    objects = root.findall("object")
    names = [obj.find("name").text.strip() for obj in objects 
      if obj.find("deleted").text.strip() == "0"]
    return names


  def get_category(self, name):
    cleaned_name = name.lower().replace("_", " ")
    try:
      return self.known_cats_rev[cleaned_name]
    except:
      return "other"


<<<<<<< HEAD
  def random_colour(self, seed=None):
    return (randint(255), randint(255), randint(255))
      

  def visualize(self, id, alpha=0.2):
    img = np.load('data/numpy/images/' + id + '.npy')
    label = np.load('data/numpy/labels/' + id + '.npy')
    shape = img.shape[:-1]
    highlight = np.zeros((*shape, 3), dtype=np.uint8)
    colour_mask = np.zeros((*shape, 3), dtype=np.uint8)
    masked_image = np.zeros((*shape, 3), dtype=np.uint8)

    colours = [self.random_colour() for _ in range(self.n_categories)]

    objects = eval(self.file_df['objects'].loc[self.file_df['id']==id][0])
    for obj in objects:
      cat = self.get_category(obj)
      index = self.sort_order(cat)
      highlight[:] = colours[index]
      select = label[:,:,index].astype(np.uint8)
      highlight = cv2.bitwise_and(highlight, highlight, mask=select)
      colour_mask += highlight    

    masked_image += cv2.addWeighted(img, alpha, colour_mask, 1 - alpha, 0)
    
    with_legend = np.zeros((shape[0], shape[1] + 60, 3), dtype=np.uint8) # pad to make room for legend (hack)
    with_legend[:,:-60] += masked_image
    plt.figure()
    plt.imshow(with_legend)    
    custom_lines = [Line2D([0], [0], color=[c/255 for c in colour], lw=4) for colour in colours]
    cat_names = [cat for cat in sl.known_cats] + ['other']
    plt.legend(custom_lines, cat_names)
    plt.show()    


def load_image(file):
  img = cv2.imread(file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img
=======

      

>>>>>>> 2143fb837b58af2288d20c39e77eea1aa3fcee57
