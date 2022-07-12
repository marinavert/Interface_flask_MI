from flask import Flask, render_template, request
import pickle, os

app = Flask(__name__, template_folder='templates', static_folder='static')

import numpy as np
from canny import canny_edge_detector

result_path = 'static/canny'


import matplotlib.pyplot as plt
import cv2
def maincanny(img, scale = 0.1): 
    from PIL import Image, ImageDraw
    #savefiles = [os.path.join(result_path, os.path.basename(f)) for f in imgfiles]
    #if not os.path.exists(result_path): 
     #   os.mkdir(result_path)
    #for i in range(len(imgfiles)): 
    input_image = Image.open(img)
    w, h = input_image.size
    newW, newH = int(scale * w), int(scale * h)
    input_image = input_image.resize((newW, newH), resample=Image.BICUBIC)
    output_image = Image.new("RGB", input_image.size)
    draw = ImageDraw.Draw(output_image)
    for x, y in canny_edge_detector(input_image):
        draw.point((x, y), (255, 255, 255))
    output_image.save(os.path.join(result_path,os.path.basename(img)))
    #print("Finished img", i)    


from segmentation import predict_img
from PIL import Image
from utils import UNet 
import torch

plate_model = 'crimson_pyramid_51.pth'
food_model = 'hearty_wood_4.pth'
plate_net = UNet(n_channels=3, n_classes=2,bilinear=False)
plate_net.load_state_dict(torch.load(plate_model, map_location=torch.device('cpu')))
food_net = UNet(n_channels=3, n_classes=2,bilinear=False)
food_net.load_state_dict(torch.load(food_model, map_location=torch.device('cpu')))

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def mainpredict(imgpath, cannypath, scale): 
    img = Image.open(imgpath)
    canny = Image.open(cannypath)

    w, h = img.size
    newW, newH = int(scale * w), int(scale * h)
    img = img.resize((newW, newH), resample=Image.BICUBIC)
    canny = canny.resize((newW, newH), resample=Image.BICUBIC)

    device='cpu'

    plate_mask = predict_img(net=plate_net,
                           full_img=canny,
                        #    scale_factor=0.5,
                           out_threshold=0, 
                           device=device)
    food_mask = predict_img(net=food_net,
                           full_img=img,
                        #    scale_factor=0.5,
                           out_threshold=0,
                           device=device)
    
    plate_out_filename = os.path.join('static/plateseg/', os.path.basename(cannypath))
    plate_result = mask_to_image(plate_mask)
    plate_result.save(plate_out_filename)

    food_out_filename = os.path.join('static/foodseg/', os.path.basename(imgpath))
    food_result = mask_to_image(food_mask)
    food_result.save(food_out_filename)

def computeScore(foodsegpath, platesegpath, inputratio): 
    import cv2 as cv
    foodimg = cv.imread(foodsegpath)
    plateimg = cv.imread(platesegpath)
    foodvolume = np.count_nonzero(foodimg)
    platevolume = np.count_nonzero(plateimg) 
    score = 100 * (1 + inputratio - foodvolume/platevolume)
    return (int(score) if score < 100 else 100)

def computeRatio(foodsegpath, platesegpath):
    import cv2 as cv
    foodimg = cv.imread(foodsegpath)
    plateimg = cv.imread(platesegpath)
    foodvolume = np.count_nonzero(foodimg)
    platevolume = np.count_nonzero(plateimg) 
    return foodvolume/platevolume

# plate_model.predict()
# food_model.predict()

# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,3)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")



@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == "POST":
        img = request.files["my_image"]
        img_path = "static/img/" + img.filename
        #print(img.filename)
        img_canny = maincanny(img_path)
        canny_path = "static/canny/" + img.filename

        mainpredict(img_path, canny_path, 0.1)
        foodseg_path = "static/foodseg/" + img.filename
        plateseg_path = "static/plateseg/" + img.filename
        score = computeScore(foodseg_path, plateseg_path, 1e-3)
        ratio = computeRatio(foodseg_path, plateseg_path)
        str_ratio = "{:.2f}".format(ratio)

    return render_template("index.html", img_path = img_path, canny_path = canny_path, foodseg_path=foodseg_path, plateseg_path=plateseg_path, score=score, ratio=str_ratio)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)

