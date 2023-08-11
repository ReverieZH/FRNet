
import random

from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, ycbcr_image, mask, image_path=None):
        assert img.size == mask.size, "image:" + image_path
        for t in self.transforms:
            img, ycbcr_image, mask = t(img, ycbcr_image, mask)
        return img, ycbcr_image, mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, ycbcr_image, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), ycbcr_image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, ycbcr_image, mask

class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, ycbcr_image, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), ycbcr_image.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)

class Edge_Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, ycbcr_image, mask, edge, image_path=None):
        assert img.size == mask.size, "image:" + image_path
        assert img.size == edge.size, "image:" + image_path
        assert img.size == ycbcr_image.size, "image:" + image_path
        assert ycbcr_image.size == mask.size, "image:" + image_path
        assert ycbcr_image.size == edge.size, "image:" + image_path
        for t in self.transforms:
            img, ycbcr_image, mask, edge = t(img, ycbcr_image, mask, edge)
        return img, ycbcr_image, mask, edge

class Edge_RandomHorizontallyFlip(object):
    def __call__(self, img, ycbcr_image, mask, edge):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), ycbcr_image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), edge.transpose(Image.FLIP_LEFT_RIGHT)
        return img, ycbcr_image, mask, edge

class Edge_Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, ycbcr_image, mask, edge):
        assert img.size == mask.size
        assert img.size == edge.size
        return img.resize(self.size, Image.BILINEAR), ycbcr_image.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.BILINEAR), edge.resize(self.size, Image.BILINEAR)