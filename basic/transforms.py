#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import itertools
import numpy as np
from skimage import transform as stf
from numpy import random, floor
from PIL import Image, ImageOps
from cv2 import erode, dilate
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from torchvision.transforms import RandomPerspective, InterpolationMode
"""
Each transform class defined here takes as input a PIL Image and returns the modified PIL Image
"""

class RandomSwirl:
    """
    Duplicate and stack image vertically
    """

    def __init__(self):
        self.max_swirl = 64
    
    def __call__(self, x):
        num_swirl = np.random.randint(1, self.max_swirl)
        self.patch_size = min(x.width, x.height)//16
        for i in range(num_swirl):
            x0, y0 = np.random.randint(0, x.width-self.patch_size), np.random.randint(0, x.height-self.patch_size)
            x1, y1 = x0 + self.patch_size, y0 + self.patch_size

            patch = x.crop((x0, y0, x1, y1))
        
            patch = stf.swirl(np.uint8(patch.convert('L')), rotation=0, strength= 3.2*(np.random.rand()-0.5), radius=self.patch_size//2)*255
            patch = Image.fromarray(patch)

            x.paste(patch.convert('RGB'), (x0, y0))
                    
        return x

class VerticalStack:
    """
    Duplicate and stack image vertically
    """

    def __init__(self):
        pass
    
    def __call__(self, x, lbl):
        x = x.transform(x.size, Image.MESH, self.generate_mesh(x), resample=Image.BICUBIC)

        lbl = '\n'.join([lbl]*2)
        return x, lbl

    def generate_mesh(self, x):
        offset = np.random.randint(x.height//3)
        offset -= offset//2
        mesh = [[[0, 0, x.width, x.height//2-offset], [0, 0, 0, x.height, x.width, x.height, x.width, 0]],
                [[0, x.height//2-offset, x.width, x.height], [0, 0, 0, x.height, x.width, x.height, x.width, 0]]]
        return mesh

class RandomVerticalPad:
    """
    Random vertical padding
    """

    def __init__(self, thres):
        self.thres = thres 

    def __call__(self, x):
        if x.height < self.thres: 
            clr = int(np.argmax(np.bincount(np.array(x).reshape(-1))))
            h_pad_max = x.height//3
            h_pad = np.random.randint(1, h_pad_max)
            y = Image.new(x.mode, (x.width, x.height+h_pad), color=(clr, clr, clr))
            y.paste(x, box=(0, np.random.randint(h_pad)))
        else:
            y = x
            
        return y

class RandomHorizontalPad:
    """
    Random horizontal padding
    """

    def __init__(self, thres):
        self.thres = thres 

    def __call__(self, x):
        if x.width < self.thres: 
            clr = int(np.argmax(np.bincount(np.array(x).reshape(-1))))
            w_pad_max = x.width//3
            w_pad = np.random.randint(1, w_pad_max)
            y = Image.new(x.mode, (x.width+w_pad, x.height), color=(clr, clr, clr))
            y.paste(x, box=(np.random.randint(w_pad), 0))
        else:
            y = x
            
        return y

class SignFlipping:
    """
    Color inversion
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return ImageOps.invert(x)


class DPIAdjusting:
    """
    Resolution modification
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        w, h = x.size
        return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)


class Dilation:
    """
    OCR: stroke width increasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(dilate(np.array(x), self.kernel, iterations=self.iterations))


class Erosion:
    """
    OCR: stroke width decreasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(erode(np.array(x), self.kernel, iterations=self.iterations))


class ElasticDistortion:
    """
    Elastic Distortion adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    def __init__(self, grid, magnitude, min_sep):

        self.grid_width, self.grid_height = grid
        self.xmagnitude, self.ymagnitude = magnitude
        self.min_h_sep, self.min_v_sep = min_sep

    def __call__(self, x):
        w, h = x.size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        shift = [[(0, 0) for x in range(horizontal_tiles)] for y in range(vertical_tiles)]

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

                sm_h = min(self.xmagnitude, width_of_square - (self.min_h_sep + shift[vertical_tile][horizontal_tile - 1][
                    0])) if horizontal_tile > 0 else self.xmagnitude
                sm_v = min(self.ymagnitude, height_of_square - (self.min_v_sep + shift[vertical_tile - 1][horizontal_tile][
                    1])) if vertical_tile > 0 else self.ymagnitude

                dx = random.randint(-sm_h, self.xmagnitude)
                dy = random.randint(-sm_v, self.ymagnitude)
                shift[vertical_tile][horizontal_tile] = (dx, dy)

        shift = list(itertools.chain.from_iterable(shift))

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for id, (a, b, c, d) in enumerate(polygon_indices):
            dx = shift[id][0]
            dy = shift[id][1]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        self.generated_mesh = generated_mesh

        return x.transform(x.size, Image.MESH, self.generated_mesh, resample=Image.BICUBIC)


class RandomTransform:
    """
    Random Transform adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    def __init__(self, val):

        self.val = val

    def __call__(self, x):
        w, h = x.size

        dw, dh = (self.val, 0) if random.randint(0, 2) == 0 else (0, self.val)

        def rd(d):
            return random.uniform(-d, d)

        def fd(d):
            return random.uniform(-dw, d)

        # generate a random projective transform
        # adapted from https://navoshta.com/traffic-signs-classification/
        tl_top = rd(dh)
        tl_left = fd(dw)
        bl_bottom = rd(dh)
        bl_left = fd(dw)
        tr_top = rd(dh)
        tr_right = fd(min(w * 3 / 4 - tl_left, dw))
        br_bottom = rd(dh)
        br_right = fd(min(w * 3 / 4 - bl_left, dw))

        tform = stf.ProjectiveTransform()
        tform.estimate(np.array((
            (tl_left, tl_top),
            (bl_left, h - bl_bottom),
            (w - br_right, h - br_bottom),
            (w - tr_right, tr_top)
        )), np.array((
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        )))

        # determine shape of output image, to preserve size
        # trick take from the implementation of skimage.transform.rotate
        corners = np.array([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ])

        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.around((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = stf.SimilarityTransform(translation=translation)
        tform = tform4 + tform
        # normalize
        tform.params /= tform.params[2, 2]

        x = stf.warp(np.array(x), tform, output_shape=output_shape, cval=255, preserve_range=True)
        x = stf.resize(x, (h, w), preserve_range=True).astype(np.uint8)

        return Image.fromarray(x)


def apply_data_augmentation(img, lbl, aug=None):
    if aug is not None:
        # Apply data augmentation
        if "random_swirl" in aug.keys() and np.random.rand() < aug["random_swirl"]["proba"]:
            img = RandomSwirl()(img)
        
        if "vertical_stack" in aug.keys() and np.random.rand() < aug["vertical_stack"]["proba"]:
            img, lbl = VerticalStack()(img, lbl)
        
        if "random_vertical_pad" in aug.keys() and np.random.rand() < aug["random_vertical_pad"]["proba"]:
            img = RandomVerticalPad(aug["random_vertical_pad"]['thres'])(img)

        if "random_horizontal_pad" in aug.keys() and np.random.rand() < aug["random_horizontal_pad"]["proba"]:
            img = RandomHorizontalPad(aug["random_horizontal_pad"]['thres'])(img)
        
        if "dpi" in aug.keys() and np.random.rand() < aug["dpi"]["proba"]:
            factor = np.random.uniform(aug["dpi"]["min_factor"], aug["dpi"]["max_factor"])
            img = DPIAdjusting(factor)(img)
        if "perspective" in aug.keys() and np.random.rand() < aug["perspective"]["proba"]:
            scale = np.random.uniform(aug["perspective"]["min_factor"], aug["perspective"]["max_factor"])
            img = RandomPerspective(distortion_scale=scale, p=1, interpolation=InterpolationMode.BILINEAR, fill=255)(img)
        elif "elastic_distortion" in aug.keys() and np.random.rand() < aug["elastic_distortion"]["proba"]:
            magnitude = np.random.randint(1, aug["elastic_distortion"]["max_magnitude"] + 1)
            kernel = np.random.randint(1, aug["elastic_distortion"]["max_kernel"] + 1)
            magnitude_w, magnitude_h = (magnitude, 1) if np.random.randint(2) == 0 else (1, magnitude)
            img = ElasticDistortion(grid=(kernel, kernel), magnitude=(magnitude_w, magnitude_h), min_sep=(1, 1))(
                img)
        elif "random_transform" in aug.keys() and np.random.rand() < aug["random_transform"]["proba"]:
            img = RandomTransform(aug["random_transform"]["max_val"])(img)
        if "dilation_erosion" in aug.keys() and np.random.rand() < aug["dilation_erosion"]["proba"]:	       
            kernel_h = np.random.randint(aug["dilation_erosion"]["min_kernel"],
                                         aug["dilation_erosion"]["max_kernel"] + 1)
            kernel_w = np.random.randint(aug["dilation_erosion"]["min_kernel"],
                                         aug["dilation_erosion"]["max_kernel"] + 1)
            if np.random.randint(2) == 0:
                img = Erosion((kernel_w, kernel_h), aug["dilation_erosion"]["iterations"])(img)
            else:
                img = Dilation((kernel_w, kernel_h), aug["dilation_erosion"]["iterations"])(img)
        if "contrast" in aug.keys() and np.random.rand() < aug["contrast"]["proba"]:
            factor = np.random.uniform(aug["contrast"]["min_factor"], aug["contrast"]["max_factor"])
            img = adjust_contrast(img, factor)
        if "brightness" in aug.keys() and np.random.rand() < aug["brightness"]["proba"]:
            factor = np.random.uniform(aug["brightness"]["min_factor"], aug["brightness"]["max_factor"])
            img = adjust_brightness(img, factor)
        if "sign_flipping" in aug.keys() and np.random.rand() < aug["sign_flipping"]["proba"]:
            img = SignFlipping()(img)
    return img, lbl
