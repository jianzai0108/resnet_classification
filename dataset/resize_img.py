from PIL import Image
import os
os.chdir('/Users/wangjian/Documents/实验室/CAD项目/resnet_classification/dataset')
for root, dirs, files in os.walk('./images/5.wardrobe'):
    for f in files:
        source = os.path.join(root, f)
        img = Image.open(source)
        img = img.resize((224, 224))
        img.save(source)
