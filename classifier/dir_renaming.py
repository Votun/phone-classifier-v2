import os
import re

def regex_correct(filename: str):
    print(filename)
    extra = re.search("_(.+?)\.", filename.split('/')[-1]).group(1)
    return filename.replace(extra, '')

for dirpath, dirs, files in os.walk("../data/mobile_images/mobile_images/"):
    print("Hello")
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = dirpath + filename
            os.rename(file_path, regex_correct(file_path))
            print(file_path)