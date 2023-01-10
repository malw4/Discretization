class Seed:
    seed_points_tab = []
    seeds_tab = []
    seeds_number = 0
    circuits_tab = []
    contours_tab = []
    is_contour = None

    def check_seeds(self, x, y, i, im_brightness):
        if (x, y) not in self.seed_points_tab:
            self.seed_points_tab.append((x, y))
            self.seeds_tab[i].append((x, y))
            if x + 1 < len(im_brightness):
                if im_brightness[x + 1, y] == 0:
                    self.check_seeds(x + 1, y, i, im_brightness)
                    self.is_contour = False
                else:
                    self.add_to_circuits(i)
                    self.is_contour = True
            else:
                self.add_to_circuits(i)
                self.is_contour = True
            if x - 1 >= 0:
                if im_brightness[x - 1, y] == 0:
                    self.check_seeds(x - 1, y, i, im_brightness)
                    self.is_contour = False
                else:
                    self.add_to_circuits(i)
                    self.is_contour = True
            else:
                self.add_to_circuits(i)
                self.is_contour = True,
            if y + 1 < len(im_brightness):
                if im_brightness[x, y + 1] == 0:
                    self.check_seeds(x, y + 1, i, im_brightness)
                    self.is_contour = False
                else:
                    self.add_to_circuits(i)
                    self.is_contour = True
            else:
                self.add_to_circuits(i)
                self.is_contour = True
            if 0 <= y - 1:
                if im_brightness[x, y - 1] == 0:
                    self.check_seeds(x, y - 1, i, im_brightness)
                    self.is_contour = False
                else:
                    self.add_to_circuits(i)
                    self.is_contour = True
            else:
                self.add_to_circuits(i)
                self.is_contour = True
            if self.is_contour:
                self.contours_tab[i].append((x, y))

    def add_to_circuits(self, i):
        self.circuits_tab[i] += 1
        self.is_contour = True
