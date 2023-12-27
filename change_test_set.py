import pandas as pd
import os
from sklearn.model_selection import train_test_split

root_directory = './data'
train_directory = root_directory + '/images/'
val_directory = root_directory + '/validate/'
all_classes = os.listdir(train_directory)

for classes in all_classes:
    new_dir = os.path.join(val_directory, classes)
    os.makedirs(new_dir, exist_ok=True)
    all_pic = os.listdir(os.path.join(train_directory, classes))

    train_pic, val_pic = train_test_split(all_pic, test_size=0.2, random_state=42)

    # Move validation images to the validation directory
    for pic in val_pic:
        src = os.path.join(train_directory, classes, pic)
        dst = os.path.join(val_directory, classes, pic)
        os.rename(src, dst)
    
    