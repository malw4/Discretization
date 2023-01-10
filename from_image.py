import math
import os
import shutil

import microstructpy as msp
import numpy as np
from matplotlib import image as mpim
from matplotlib import pyplot as plt

from Seed import Seed


class Point:
    def __init__(self, x, y, payload=None):
        self.x, self.y = x, y
        self.payload = payload

    def __repr__(self):
        return '{}: {}'.format(str((self.x, self.y)), repr(self.payload))

    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.x, self.y)

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)


def distance(point1, point2):
    if point1 != point2:
        x = point1[0] - point2[0]
        y = point1[1] - point2[1]
    else:
        x = 0.01
        y = 0.01
    return math.sqrt(x * x + y * y)


class Voronoi:
    pmesh = 0
    tmesh = 0
    im_brightness = 0
    ziarna = None
    width = 0
    height = 0

    def __init__(self, image_basename):
        # Read in image
        image_path = os.path.dirname(__file__)
        image_filename = os.path.join(image_path, image_basename)
        image = mpim.imread(image_filename)
        self.im_brightness = image[:, :, 0]
        self.ziarna = Seed()

        # Bin the pixels
        br_bins = [0.00, 0.50, 1.00]
        bin_nums = np.zeros_like(self.im_brightness, dtype='int')
        for i in range(len(br_bins) - 1):
            lb = br_bins[i]
            ub = br_bins[i + 1]
            mask = np.logical_and(self.im_brightness >= lb, self.im_brightness <= ub)
            bin_nums[mask] = i

        # Define the phases
        phases = [{'name': str(self.ziarna.seeds_number), 'color': c, 'material_type': 'amorphous', 'shape': 'ellipse'}
                  for c in ('C0', 'C1')]

        # Create the polygon mesh
        m, n = bin_nums.shape
        self.width = m
        self.height = n
        x = np.arange(n + 1).astype('float')
        y = m + 1 - np.arange(m + 1).astype('float')
        xx, yy = np.meshgrid(x, y)
        pts = np.array([xx.flatten(), yy.flatten()]).T
        kps = np.arange(len(pts)).reshape(xx.shape)

        n_facets = 2 * (m + m * n + n)
        n_regions = m * n
        facets = np.full((n_facets, 2), -1)
        regions = np.full((n_regions, 4), 0)
        region_phases = np.full(n_regions, 0)
        facet_top = np.full((m, n), -1, dtype='int')
        facet_bottom = np.full((m, n), -1, dtype='int')
        facet_left = np.full((m, n), -1, dtype='int')
        facet_right = np.full((m, n), -1, dtype='int')

        self.create_seeds(m, n)
        print(self.ziarna.seeds_tab)

        k_facets = 0
        k_regions = 0
        for i in range(m):
            for j in range(n):
                kp_top_left = kps[i, j]
                kp_bottom_left = kps[i + 1, j]
                kp_top_right = kps[i, j + 1]
                kp_bottom_right = kps[i + 1, j + 1]

                # left facet
                if facet_left[i, j] < 0:
                    fnum_left = k_facets
                    facets[fnum_left] = (kp_top_left, kp_bottom_left)
                    k_facets += 1

                    if j > 0:
                        facet_right[i, j - 1] = fnum_left
                else:
                    fnum_left = facet_left[i, j]

                # right facet
                if facet_right[i, j] < 0:
                    fnum_right = k_facets
                    facets[fnum_right] = (kp_top_right, kp_bottom_right)
                    k_facets += 1

                    if j + 1 < n:
                        facet_left[i, j + 1] = fnum_right
                else:
                    fnum_right = facet_right[i, j]

                # top facet
                if facet_top[i, j] < 0:
                    fnum_top = k_facets
                    facets[fnum_top] = (kp_top_left, kp_top_right)
                    k_facets += 1

                    if i > 0:
                        facet_bottom[i - 1, j] = fnum_top
                else:
                    fnum_top = facet_top[i, j]

                # bottom facet
                if facet_bottom[i, j] < 0:
                    fnum_bottom = k_facets
                    facets[fnum_bottom] = (kp_bottom_left, kp_bottom_right)
                    k_facets += 1

                    if i + 1 < m:
                        facet_top[i + 1, j] = fnum_bottom
                else:
                    fnum_bottom = facet_bottom[i, j]

                # region
                region = (fnum_top, fnum_left, fnum_bottom, fnum_right)
                regions[k_regions] = region
                region_phases[k_regions] = bin_nums[i, j]
                k_regions += 1
        '''
        self.pmesh = msp.meshing.PolyMesh(pts, facets, regions,
                                          seed_numbers=range(n_regions),
                                          phase_numbers=region_phases)
        print("pmesh")
        # Create the triangle mesh

        self.tmesh = msp.meshing.TriMesh.from_polymesh(self.pmesh, phases=phases, min_angle=20)

        self.tmesh.write('outputStruct', 'abaqus', polymesh=self.pmesh)
        # self.tmesh.write('outputStruct','abaqus',seeds,self.pmesh,phase_numbers=region_phases)
        print("tmesh")

        # Plot triangle mesh
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.add_axes(ax)

        fcs = [phases[region_phases[r]]['color'] for r in self.tmesh.element_attributes]
        self.tmesh.plot(facecolors=fcs, edgecolors='k', lw=0.2)

        plt.axis('square')
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
        plt.axis('off')

        # Save plot and copy input file
        plot_basename = 'from_image/trimesh.png'
        file_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(file_dir, plot_basename)
        dirs = os.path.dirname(filename)

        if not os.path.exists(dirs):
            os.makedirs(dirs)

        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        shutil.copy(image_filename, dirs)
        '''

    def create_seeds(self, m, n):
        for x in range(m - 1):
            for y in range(n - 1):
                if self.im_brightness[x, y] == 0:
                    if (x, y) not in self.ziarna.seed_points_tab:
                        self.ziarna.seeds_tab.append([])
                        self.ziarna.contours_tab.append([])
                        self.ziarna.circuits_tab.append(0)
                        self.ziarna.check_seeds(x, y, self.ziarna.seeds_number, self.im_brightness)
                        self.ziarna.seeds_number += 1

    def compute_factors(self):
        seeds_number = 0
        for seed in self.ziarna.seeds_tab:
            l = self.ziarna.circuits_tab[seeds_number]
            s = len(seed)
            sum_l = 0
            sum_r2 = 0
            sum_d = 0
            sum_d2 = 0
            rmax = 1
            rmin = self.width * self.height
            lh = self.get_lh(seeds_number)
            lv = self.get_lv(seeds_number)
            lmax = self.get_lmax(seeds_number)
            center_of_gravity = self.get_center_of_gravity(seeds_number)
            sum_l, sum_r2 = self.get_sum_variables(center_of_gravity, seed, seeds_number, sum_l, sum_r2)
            rmax, rmin, sum_d, sum_d2 = self.get_more_variables(center_of_gravity, rmax, rmin, seeds_number,
                                                                sum_d, sum_d2)
            Rw1 = 2 * math.sqrt(s / math.pi)
            Rw2 = l / math.pi
            Rw3 = (l / (2 * math.sqrt(math.pi * s))) - 1
            Rw4 = s / (math.sqrt(2 * math.pi * sum_r2))
            Rw5 = math.pow(s, 3) / math.pow(sum_l, 2)
            Rw6 = math.sqrt(math.pow(sum_d, 2) / (len(self.ziarna.contours_tab[seeds_number]) * sum_d2 - 1))
            Rw7 = rmin / rmax
            Rw8 = lmax / l
            Rw9 = (2 * math.sqrt(math.pi * s)) / l
            Rw10 = lh / lv
            print("Seed nr " + str(seeds_number + 1))
            print("Rw1 = " + str(Rw1))
            print("Rw2 = " + str(Rw2))
            print("Rw3 = " + str(Rw3))
            print("Rw4 = " + str(Rw4))
            print("Rw5 = " + str(Rw5))
            print("Rw6 = " + str(Rw6))
            print("Rw7 = " + str(Rw7))
            print("Rw8 = " + str(Rw8))
            print("Rw9 = " + str(Rw9))
            print("Rw10 = " + str(Rw10))
            seeds_number += 1

    def get_sum_variables(self, center_of_gravity, seed, seeds_number, sum_l, sum_r2):
        for point in seed:
            sum_l += self.min_contour_dist(point, seeds_number, self.width, self.height)
            sum_r2 += math.pow(distance(point, center_of_gravity), 2)
        return sum_l, sum_r2

    def get_more_variables(self, center_of_gravity, r_max, r_min, seeds_number, sum_d, sum_d2):
        for point in self.ziarna.contours_tab[seeds_number]:
            point_dist = distance(point, center_of_gravity)
            sum_d += point_dist
            sum_d2 += math.pow(point_dist, 2)
            if point_dist < r_min:
                r_min = point_dist
            if point_dist > r_max:
                r_max = point_dist
        return r_max, r_min, sum_d, sum_d2

    def min_contour_dist(self, point, i, m, n):
        dist_min = m * n
        for point_in_contour in self.ziarna.contours_tab[i]:
            current_dist = distance(point, point_in_contour)
            if current_dist < dist_min:
                dist_min = current_dist
        return dist_min

    def get_lmax(self, i):
        lmax = 0
        for point1 in self.ziarna.contours_tab[i]:
            for point2 in self.ziarna.contours_tab[i]:
                current_dist = distance(point1, point2)
                if current_dist > lmax:
                    lmax = current_dist
        return lmax

    def get_center_of_gravity(self, i):
        x_sum = 0
        y_sum = 0
        for point in self.ziarna.seeds_tab[i]:
            x_sum += point[0]
            y_sum += point[1]
        return x_sum / len(self.ziarna.seeds_tab[i]), y_sum / len(self.ziarna.seeds_tab[i])

    def get_lh(self, i):
        lh = 0
        for point in self.ziarna.contours_tab[i]:
            points_in_x = []
            for point1 in self.ziarna.contours_tab[i]:
                if point1[0] == point[0]:
                    points_in_x.append(point1)
            for point_in_x in points_in_x:
                current_dist = distance(point, point_in_x)
                if current_dist > lh:
                    lh = current_dist
        return lh

    def get_lv(self, i):
        lv = 0
        for point in self.ziarna.contours_tab[i]:
            points_in_y = []
            for point1 in self.ziarna.contours_tab[i]:
                if point1[1] == point[1]:
                    points_in_y.append(point1)
            for point_in_y in points_in_y:
                current_dist = distance(point, point_in_y)
                if current_dist > lv:
                    lv = current_dist
        return lv
