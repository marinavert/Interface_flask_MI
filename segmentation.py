import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def preprocess(pil_img):
    w, h = pil_img.size
    # newW, newH = int(scale * w), int(scale * h)
    newW, newH = 256, 256
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
    img_ndarray = np.asarray(pil_img)

    if img_ndarray.ndim == 2:
        img_ndarray = img_ndarray[np.newaxis, ...]
    else:
        img_ndarray = img_ndarray.transpose((2, 0, 1))

    img_ndarray = img_ndarray / 255


    return img_ndarray

def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(preprocess(full_img))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
