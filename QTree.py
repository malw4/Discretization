import numpy as np
from PIL import Image, ImageDraw

MAX_DEPTH = 7
DETAIL_THRESHOLD = 20
SIZE_MULT = 1


def average_colour(image):
    image_arr = np.asarray(image)

    # get average of whole image
    if not len(image_arr):
        avg_color = image_arr
    else:
        avg_color_per_row = np.average(image_arr, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
    return int(avg_color[0]), int(avg_color[1]), int(avg_color[2])


def weighted_average(hist):
    total = sum(hist)
    error = value = 0

    if total > 0:
        value = sum(i * x for i, x in enumerate(hist)) / total
        error = sum(x * (value - i) ** 2 for i, x in enumerate(hist)) / total
        error = error ** 0.5

    return error


def get_detail(hist):
    red_detail = weighted_average(hist[:256])
    green_detail = weighted_average(hist[256:512])
    blue_detail = weighted_average(hist[512:768])

    detail_intensity = red_detail * 0.2989 + green_detail * 0.5870 + blue_detail * 0.1140

    return detail_intensity


class Quadrant:
    def __init__(self, image, bbox, depth):
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False

        # crop image to quadrant size
        image = image.crop(bbox)
        hist = image.histogram()

        self.detail = get_detail(hist)
        self.color = average_colour(image)

    def split(self, image):
        left, top, width, height = self.bbox

        # middle coords of bbox
        mid_x = left + (width - left) / 2
        mid_y = top + (height - top) / 2

        # split into new quadrants
        Q1 = Quadrant(image, (left, top, mid_x, mid_y), self.depth + 1)
        Q2 = Quadrant(image, (mid_x, top, width, mid_y), self.depth + 1)
        Q3 = Quadrant(image, (left, mid_y, mid_x, height), self.depth + 1)
        Q4 = Quadrant(image, (mid_x, mid_y, width, height), self.depth + 1)

        self.children = [Q1, Q2, Q3, Q4]


class QuadTree:
    def __init__(self, image):
        self.root = None
        self.width, self.height = image.size

        self.max_depth = 0

        self.start(image)

    def start(self, image):
        self.root = Quadrant(image, image.getbbox(), 0)

        self.build(self.root, image)

    def build(self, root, image):
        if root.depth >= MAX_DEPTH or root.detail <= DETAIL_THRESHOLD:
            if root.depth > self.max_depth:
                self.max_depth = root.depth

            # quadrant is a leaf so stop recursing
            root.leaf = True
            return

            # split quadrant if there is too much detail
        root.split(image)

        for children in root.children:
            self.build(children, image)

    def create_image(self, custom_depth, show_lines=False):
        # blank image
        image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.width, self.height), (0, 0, 0))

        leaf_quadrants = self.get_leaf_quadrants(custom_depth)

        # rectangle size of quadrant for each leaf quadrant
        for quadrant in leaf_quadrants:
            if show_lines:
                draw.rectangle(quadrant.bbox, quadrant.color, outline=(0, 0, 0))
            else:
                draw.rectangle(quadrant.bbox, quadrant.color)
        return image

    def get_leaf_quadrants(self, depth):
        if depth > self.max_depth:
            raise ValueError('A depth is larger than the trees depth was given')

        quandrants = []

        # search recursively down the quadtree
        self.recursive_search(self, self.root, depth, quandrants.append)

        return quandrants

    def recursive_search(self, tree, quadrant, max_depth, append_leaf):
        # append if quadrant is a leaf
        if quadrant.leaf == True or quadrant.depth == max_depth:
            append_leaf(quadrant)

        # otherwise recurse
        elif quadrant.children is not None:
            for child in quadrant.children:
                self.recursive_search(tree, child, max_depth, append_leaf)

    def save_final_image(self, show_lines=False):
        end_product_image = self.create_image(self.max_depth, show_lines=show_lines)
        end_product_image.save("Qtree_Struct.png")
