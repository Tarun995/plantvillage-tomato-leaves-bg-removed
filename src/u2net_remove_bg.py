import torch
import cv2
import numpy as np
from u2net.u2net import U2NET

def load_model(model_path):
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net

def u2net_remove_background_from_numpy(net, image_np):
    image = cv2.resize(image_np, (320, 320)) / 255.0
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()

    with torch.no_grad():
        d1, *_ = net(image)
        mask = d1[0][0].numpy()
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = (mask > 0.2).astype(np.uint8) * 255

    result = cv2.bitwise_and(image_np, image_np, mask=mask)
    return result
