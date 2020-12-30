import numpy as np
from sklean import linear_model

def add_square_feature(X):
    X = np.concatenate([(X**2).reshape(-1,1), X], axis=1)
    return X

class LaneBoundary:
    def __init__(self, xpoints, ypoints):
        try:
            self.__points_x = xpoints
            self.__points_y = ypoints
            self.__fit_coeffs = []
            self.__fit_coeffs_meters = []
            self.__location = 0
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

    @property
    def location(self):
        return self.__location

    @property
    def func(self):
        return self.__func

    def fit_point(self, ):
        try:
            # Fit a second order polynomial to each using `np.polyfit`
            self.__fit_coeffs = np.polyfit(self.points_y, self.points_x, 2)

            # Robustly fit linear model with RANSAC algorithm
            # ransac = linear_model.RANSACRegressor()
            # ransac.fit(add_square_feature(self.points_y), self.points_x)
            self.__func = np.poly1d(self.__fit_coeffs)
            return True
        except Exception as inst:
            print(self.points_x.shape)
            print(self.points_y.shape)
            print(type(inst), inst.args, inst)
            return False

    def get_location(self):
        """
            Calculate the location at bottom of the image
        """
        self.location = self.func(0)
