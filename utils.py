from PIL import Image
import numpy as np
from torchvision import transforms
import math

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

def prepare_image(image):
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

class PrepareImg:
    def __init__(self, resizeHW):
        transform_list = []
        transform_list.append(transforms.Resize((resizeHW,resizeHW)))
        transform_list += [transforms.ToTensor(),]
        self.transform = transforms.Compose(transform_list)

    def prepare_image(self, image):
        image = self.transform(image)
        return image.unsqueeze(0)

def load_img(img_path, robustness_distortion=None, pct_shift=1, ref=True):
    if robustness_distortion == "tra": # translation
        h, w = Image.open(img_path).size
        h_offset = np.ceil((pct_shift/100.)*h) # horizontal_shift_offset
        shifted_h = h-h_offset
        noShift = (0, 0, shifted_h, w) # left, top, right, bottom
        shifted = (h_offset, 0, h, w) # left, top, right, bottom
        if ref:
            return Image.open(img_path).crop(noShift).convert("RGB")
        else:
            return Image.open(img_path).crop(shifted).convert("RGB")
    elif robustness_distortion == "sca": # scale
        h, w = Image.open(img_path).size
        h_offset = int(round_up_to_even((pct_shift/100.)*h))
        w_offset = int(round_up_to_even((pct_shift/100.)*w))
        new_h = h + h_offset
        new_w = w + w_offset
        center = (new_h // 2, new_w // 2)
        left, top, right, bottom = center[0]-int(np.ceil(h/2)), center[1]-int(np.ceil(w/2)), center[0]+int(np.ceil(h/2)), center[1]+int(np.ceil(w/2))
        if ref:
            return Image.open(img_path).convert("RGB")
        else:
            return Image.open(img_path).resize((new_h, new_w), Image.Resampling.BILINEAR).crop((left, top, right, bottom)).resize((h, w), Image.Resampling.BILINEAR).convert("RGB")
    elif robustness_distortion == "rot": # rotate
        if ref:
            return Image.open(img_path).convert("RGB")
        else:
            return Image.open(img_path).rotate(-pct_shift).convert("RGB")
    else:
        return Image.open(img_path).convert("RGB")

def get_indxs_sliding_window(length=3840, window_size=224, stride=200):
    indxs = np.arange(0, length-224, stride)
    # Check if indxs is empty to avoid IndexError
    if indxs.size > 0:
        if indxs[-1]+2*window_size > length:
            indxs = np.append(indxs, [length-window_size])
        else:
            indxs = np.append(indxs, [indxs[-1]+window_size, length-window_size])
    else:
        # If indxs is empty, directly append the final window position
        indxs = np.array([length - window_size])
    return indxs