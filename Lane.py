import numpy as np

class Lane:
    def __init__(self, xpoints, ypoints):
        try:
            self.__points_x = xpoints
            self.__points_y = ypoints
            self.__fit_coeffs = []
            self.__fit_coeffs_meters = []
        except:
            print(xpoints, ypoints)
        return
    
    @property
    def fit_coeffs(self):
        return self.__fit_coeffs

    @property
    def fit_coeffs_meters(self):
        return self.__fit_coeffs_meters

    @property
    def points_x(self):
        return self.__points_x

    @property
    def points_y(self):
        return self.__points_y

    def fit_point(self):
        try:
            # Fit a second order polynomial to each using `np.polyfit`
            self.__fit_coeffs = np.polyfit(self.points_y, self.points_x, 2)
            return True
        except Exception as inst:
            print(self.points_x.shape)
            print(self.points_y.shape)
            print(type(inst), inst.args, inst)
            return False
