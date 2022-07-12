from flask import Flask, render_template, request
import pickle, os

app = Flask(__name__)


from math import sqrt, atan2, pi
import numpy as np

result_path = 'static/canny'

def canny_edge_detector(input_image):
    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height

    # Transform the image to grayscale
    grayscaled = compute_grayscale(input_pixels, width, height)

    # Blur it to remove noise
    blurred = compute_blur(grayscaled, width, height)

    # Compute the gradient
    gradient, direction = compute_gradient(blurred, width, height)

    # Non-maximum suppression
    filter_out_non_maximum(gradient, direction, width, height)

    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 20, 25)

    return keep


def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale


def compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = np.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred


def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx**2 + magy**2)
                direction[x, y] = atan2(magy, magx)
    return gradient, direction


def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0


def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)


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


def mainpredict(imgpath, cannypath): 
    img = Image.open(imgpath)
    canny = Image.open(cannypath)

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
    print(foodvolume, platevolume)
    score = 100 * (1 + inputratio - foodvolume/platevolume)
    return score

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

        mainpredict(img_path, canny_path)
        foodseg_path = "static/foodseg/" + img.filename
        plateseg_path = "static/plateseg/" + img.filename
        score = computeScore(foodseg_path, plateseg_path, 0.1)
        ratio = computeRatio(foodseg_path, plateseg_path)

    return render_template("index.html", img_path = img_path, canny_path = canny_path, foodseg_path=foodseg_path, plateseg_path=plateseg_path, score=score, ratio=ratio)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)

