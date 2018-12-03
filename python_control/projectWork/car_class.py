#from AirSimClient import *
from pyquaternion import Quaternion
import numpy as np
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])
def radToDegree(tmp):
    return tmp/np.pi*180
def degreeToRad(tmp):
    return tmp*np.pi/180
def maxValue(regulate,max_value):
    if regulate>max_value:
        regulate=max_value
    if regulate<-max_value:
        regulate=-max_value
    return regulate
def angularDiff(a,b):
    diff = a - b
    if (diff < -np.pi):
        diff = diff + np.pi * 2
    if (diff > np.pi):
        diff = diff - np.pi * 2
    return diff
Kv = 0.5
Kh = 4.
Kacell = 10
saved_steering = 0

class CarPControl:
    def __init__(self):
        self.client = CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = CarControls()
        self.client.reset()
        self.saved_steering=0

    def setPos(self,x, y):#has to be queed every step
        car_state = self.client.getCarState()
        car_pos = self.client.simGetPose()
        my_quaternion = Quaternion(np.asarray(
            [car_pos.orientation.x_val, car_pos.orientation.y_val, car_pos.orientation.z_val,
             car_pos.orientation.w_val]))
        tetta_car = rotationMatrixToEulerAngles(my_quaternion.rotation_matrix)[0]
        # print(rotationMatrixToEulerAngles(my_quaternion.rotation_matrix))

        # tetta_wanted = math.atan((wanted_pos[1]-car_pos.position.y_val)/(wanted_pos[0]-car_pos.position.x_val))
        tetta_wanted = np.arctan2((y - car_pos.position.y_val), (x - car_pos.position.x_val))
        # print(tetta_wanted)
        # print(tetta_car)
        gamma = Kh * (angularDiff(tetta_wanted, tetta_car))
        # print(gamma)
        gamma = maxValue(gamma, 0.5)

        self.car_controls.steering = gamma
        self.saved_steering=gamma
        v_wanted = Kv * np.sqrt(
            (y - car_pos.position.y_val) * (y - car_pos.position.y_val) + (
                        x - car_pos.position.x_val) * (x - car_pos.position.x_val))
        v_wanted = maxValue(v_wanted, 10.)
        accell_in = Kacell * (v_wanted - car_state.speed)
        accell_in = maxValue(accell_in, 1)
        self.car_controls.throttle = accell_in
        self.car_controls.is_manual_gear = False;
        if (accell_in < 0):
            self.car_controls.is_manual_gear = True;
            self.car_controls.manual_gear = -1
        # print(accell_in)
        self.client.setCarControls(self.car_controls)


    def getPos(self):
        pos=self.client.simGetPose()
        return [pos.position.x_val,pos.position.y_val]

    def getImage(self):
        responses = self.client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

        # reshape array to 4 channel image array H X W X 4
        mg_rgba = img1d.reshape(response.height, response.width, 4)
        return mg_rgba
    def getSteeringAngle(self):
        return self.saved_steering
    def quit(self):
        self.client.enableApiControl(False)