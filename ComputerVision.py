import numpy as np
import cv2
import math


class Computer_vision:
    COLORS = ["green", "pink"]
    COLOR_BOUNDS = {
        "green": [np.array([50, 50, 90]), np.array([85, 200, 255])],
        "pink": [np.array([135, 20, 90]), np.array([168, 200, 255])],
    }
    def __init__(self):
        self.bgr_image = None
        self.hsv_image = None
        self.point_cloud = None
        self.max_u = 640
        self.max_v = 480
        self.max_object_size = 1000000
        self.min_object_size = 500
        self.color_masks = {"green": None, "pink": None}
        self.contours = {"green": None, "pink": None}

    def _click_data(self, event, x, y, flags, param):
        """
        upon calling cv2.setMouseCallback('NAME OF WINDOW', self._click_data); this function is set to be called upon sensing an event.
         This function than checks weather the event is left mouse button click and prints useful information about the clicked pixel.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            print("[U, V]:",[x, y],"HSV:", self.hsv_image[y, x])
    def update_image(self, bgr_image):
        """
        Updates color image and point cloud. Also calculates coresponding hsv image. Call everytime new images are meant to be analyzed.
        :param bgr_image:
        :param point_cloud:
        """
        self.bgr_image = bgr_image
        self.hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        self.update_color_masks()
        self.update_contours()

    def display_bgr_image(self):
        """
        Displays color image.
        """
        cv2.imshow("BGR", self.bgr_image)
        cv2.setMouseCallback('BGR', self._click_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False


    def display_hsv_img(self):
        """
        Displays hsv image as bgr. (in the wierd way).
        """
        cv2.imshow("HSV", self.hsv_image)
        cv2.setMouseCallback('HSV', self._click_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False

    def display_color_masks(self, *colors):
        """
        Combines colormasks into one and displays it.
        :param colors: A list of colors, where colors are sring names. for example:  "red", "blue", ....
        """
        final_mask = self.color_masks[colors[0]]
        if len(colors) > 1:
            for color in colors[1:]:
                final_mask = final_mask | self.color_masks[color]
        #print(type(final_mask), final_mask.dtype)
        #print(type(self.bgr_image), self.bgr_image.dtype)
        masked_image = cv2.bitwise_and(self.bgr_image, self.bgr_image, mask=final_mask)
        cv2.imshow("Masked BGR", final_mask)
        #cv2.imshow("Masked BGR", masked_image)
        cv2.setMouseCallback('Masked BGR', self._click_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False


    def get_color_mask(self, color):
        """
        :param color: string name of desired color. For example: "red". color must be included in self.COLORS.
        :return: A 2d uint8 numpy array where 1 corresponds to color present on pixe and 0 means otherwise.
        """
        if color == "red":
            mask1 = cv2.inRange(self.hsv_image, self.COLOR_BOUNDS[color][0], self.COLOR_BOUNDS[color][1])
            mask2 = cv2.inRange(self.hsv_image, self.COLOR_BOUNDS[color][2], self.COLOR_BOUNDS[color][3])
            mask = mask1 | mask2
        else:
            mask = cv2.inRange(self.hsv_image, self.COLOR_BOUNDS[color][0], self.COLOR_BOUNDS[color][1])
        return mask

    def get_mask_stripes(self, divide_width = 10):
        shape = list(self.bgr_image.shape)[:2]
        mask = np.ones(shape, np.uint8)
        #print(shape)
        for c in range(1, shape[1]//divide_width + 1):
            column = c*divide_width
            mask[:, column] = np.zeros(shape[0], np.uint8)
        return mask

    def get_mask_no_floor(self, height):
        shape = list(self.bgr_image.shape)[:2]
        mask_top = np.ones((shape[0] - height, shape[1]), np.uint8)
        mask_bottom = np.zeros((height, shape[1]), np.uint8)
        mask = np.concatenate((mask_top,mask_bottom),axis=0)
        return mask

    def update_color_masks(self):
        """
        Creates a new mask for each color in self.COLORS.
        """
        for color in self.COLORS:
            self.color_masks[color] = self.get_color_mask(color)

    def update_contours(self):
        """
        Generates countours and filters them by size.
        Desirable contours are then placed in self.contours according to their color.
        """
        for color in self.COLORS:
            filtered_contours = []
            contours, hierarchy = cv2.findContours(self.color_masks[color], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if self.min_object_size < cv2.contourArea(contour) < self.max_object_size:
                    filtered_contours.append(contour)
            self.contours[color] = filtered_contours



    def display_contours(self, *colors):
        """
        Displays contours according to selected colors.
        :param colors: A list of colors, where colors are sring names. for example:  "red", "blue", ....
        """
        img = self.bgr_image
        for color in colors:
            cv2.drawContours(img, self.contours[color], -1, (0, 255, 0), 1)
        cv2.imshow("Contours on BGR", img)
        cv2.setMouseCallback('Contours on BGR', self._click_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False

    def display_acc(self, center, origin, vect,  *colors):
        """
        Displays contours according to selected colors.
        :param colors: A list of colors, where colors are sring names. for example:  "red", "blue", ....
        """
        img = self.bgr_image
        for color in colors:
            cv2.drawContours(img, self.contours[color], -1, (0, 255, 0), 1)
        acc_vect = self.relative_to_absolute_position(center, origin, vect)
        img = cv2.arrowedLine(img, origin.astype(int), acc_vect.astype(int), (0, 0, 255), 4, tipLength = 0.5)
        cv2.imshow("acceleartion", img)
        cv2.setMouseCallback('acceleartion', self._click_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False

    @staticmethod
    def _clockwise_angle_between_vectors(vect1: np.ndarray, vect2: np.ndarray) -> float:
        """
        :param vect1: main vector. (use Robot.direction here)
        :param vect2: secondary vector
        :return: angle between 0° and 360° in clockwise direction
        """
        ang1 = np.arctan2(*vect1[::-1])
        ang2 = np.arctan2(*vect2[::-1])
        angle = np.rad2deg(ang1 - ang2) % 360
        return angle
    @staticmethod
    def _normalize_vector(vect: np.ndarray) -> np.ndarray:
        norm = vect / np.linalg.norm(vect)
        return norm

    @staticmethod
    def _rotate_vector(vect: np.ndarray,
                       angle: float) -> np.ndarray:  # angle in degrees, kladny uhel = proti smeru hod. rucicek
        """
        Takes a vector and rotates it by angle.
        :param vect: any nonzero 2d vector
        :param angle: in degrees preferably between 180° and -180°
        :return: rotated vector
        """
        theta = np.deg2rad(-angle)
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        return (np.dot(rot, vect))
    def relative_to_absolute_position(self, center, origin, vect):
        base_vector = np.array([0, 1])
        angle = self._clockwise_angle_between_vectors(base_vector, origin - center)
        rotated_position = self._rotate_vector(vect, angle)
        absolute_position = rotated_position + origin
        return absolute_position
    def get_color_position(self, color):
        if len(self.contours[color]) > 0:
            color_contour = max(self.contours[color], key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(color_contour)
            return np.array([x, y])
        else:
            return np.array([0, 0])

    def get_list_of_objects(self):
        objects = []
        self.update_color_masks()
        for color in self.COLORS + ["grey"]:
            color_contours = self.contours[color]
            if color_contours != None:
                for index in range(len(color_contours)):
                    position = self.get_position_from_contour(color_contours, index)
                    position += np.array([0.06, 0.0], dtype="float64") #x, y
                    objects.append([color, position])
        return objects

if __name__ == "__main__":

    #np.set_printoptions(threshold=sys.maxsize)
    computer_vision = Computer_vision()
    cloud = np.load("cloud1.npy", allow_pickle=True)
    bgr_image = np.load("color1.npy", allow_pickle=True)
    computer_vision.update_image(bgr_image, cloud)
    computer_vision.update_color_masks()
    #computer_vision.display_pc_img()
    computer_vision.update_contours()
    #computer_vision.display_color_masks("red")
    computer_vision.display_contours()
    #computer_vision.display_bgr_image()
    #computer_vision.get_mask_stripes()
    #computer_vision.get_position_from_contour(computer_vision.contours["purple"], 0)
    print(computer_vision.get_list_of_objects())

    #computer_vision.display_contours("purple", "red", "green", "blue", "yellow")
    #computer_vision.display_contours("yellow")
    #computer_vision.update_connected_components()
    #computer_vision.display_color_masks("purple", "green", "red", "yellow", "blue")
    computer_vision.display_color_masks("yellow")

    #computer_vision.display_connected_components("purple", "green", "red", "yellow", "blue")

    #copmputer_vision.display_pc_img()
    #objects = copmputer_vision.get_list_of_objects()
    #print(objects)