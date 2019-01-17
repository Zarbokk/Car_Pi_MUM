# !/usr/bin/env python

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
import numpy as np
import message_filters
import rospy
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from tracking_performance_class import tracking_red_dots
import invModelControlROS as imcr
#from invModelControlforROS import invModelControl
rospy.init_node('publisher', anonymous=True)
pub = rospy.Publisher('car_input_03', PointStamped, queue_size=1)
rate = rospy.Rate(20)  # Frequenz der Anwendung
tracker= None
inverse_model=None
ueberholen = False
linie_folgen = True
fahrzeug_folgen = False
x_0 = 326
y_0 = 396
x_1 = 400
y_1 = 398
alpha_car = 0
x_car = 0
y_car = 0
angle_03=0
time_overtake=0

speed=1.1
timer_always = 0

states = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # np.array([x_b, y_b, betha_b, psi_b, phi_b, x_f, y_f, betha_f, psi_f, phi_f])

u_f = np.array([0, 0])

u_b = np.array([0, 0])
t_range = 1 / 20.0


def transform_to_opencv(image):
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)
    return image


def tiefpass(x, x_old, rate=0.5):
    return x * (1 - rate) + x_old * rate


def cubic_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def maxValue(regulate, max_value):
    if regulate > max_value:
        regulate = max_value
    if regulate < -max_value:
        regulate = -max_value
    return float(regulate)


def controller_verfolgen(x, y, alpha):
    # print("distance x:", x, "distance y:", y, alpha * 180 / np.pi, distance)
    x = x - 350

    a = -y / x ** 3 + np.tan(alpha) / x ** 2
    b = y / x ** 2 - a * x
    y_pos = cubic_function(x / 2, a, b, 0, 0)

    theta_wanted = np.arctan2(y_pos, x / 2)
    # print(x / 2, y_pos, theta_wanted * 180 / np.pi)
    # print(theta_wanted)
    # print(theta_car)
    gamma = 1 * (theta_wanted)
    gamma = maxValue(gamma * 180 / 3.14159, 29)
    # print(gamma)
    steering = gamma

    # saved_steering = gamma
    v_wanted = 1 * (x / 2 - 50)
    # v_wanted = maxValue(v_wanted, 4095)
    # accell_in = Kacell * (v_wanted - speed_car)
    accell_in = maxValue(9 * v_wanted, 4000)
    return accell_in, steering


def overtake(imu_03_sub, car_f_sub, imu_10_sub,inverse_model):
    global angle_03,timer_always,speed,states,u_b,u_f
    scaling_imu_angle = 17063036.0 / 4.0 / 360.0 * 5
    angle_03 = angle_03 + imu_03_sub.angular_velocity.z
    v = speed

    initialdelaytime = 1
    middledelaytime = 1

    if timer_always < initialdelaytime:
        v = speed
        delta = 0
    else:
        if timer_always > inverse_model.trajectory.specifics.T + initialdelaytime + middledelaytime:
            if states[0] > states[5]:
                if not inverse_model.T0:
                    print("init")
                    inverse_model.T0 = timer_always
                    # inverse_model.trajectory.setSpecifics([v,-inverse_model.trajectory.specifics.W])
                    inverse_model.trajectory.updateVsoll(speed)
                print("in seconds:",timer_always - inverse_model.T0)
                v, delta, psi = inverse_model.carInput(timer_always - inverse_model.T0)
                delta = -delta
                psi = -psi
                print(delta)
                if timer_always > 1.5 * (inverse_model.trajectory.specifics.T + inverse_model.T0):
                    v = 0
                    states[0] = 0
                    states[1] = 0

            else:
                delta = 0
                psi = 0
        elif timer_always > inverse_model.trajectory.specifics.T + initialdelaytime and timer_always < inverse_model.trajectory.specifics.T + initialdelaytime + middledelaytime:
            delta = 0
            psi = 0
        else:
            v, delta, psi = inverse_model.carInput(timer_always - initialdelaytime)
        psi = -psi * 180 / 3.14159
        delta = -delta
        error = psi - angle_03 / scaling_imu_angle
        # inverse_correction = inverse_model.trajectoryControler(error, 1.5 * 20) / 180 * 3.14159
        inverse_correction = error * 1 / 180.0 * 3.14159
        delta = delta + inverse_correction
        # print(delta, inverse_correction)
        # delta=np.pi/4
        u_b[0] = v
        u_b[1] = delta - inverse_correction
        # print(u_b[1])
        #print(t_range)
        #print(type(t_range))
        #print(states)
        #print(u_b)
        #print(u_f)
        yback = inverse_model.simulateModel(states, t_range, model="discrete", ub=u_b, uf=u_f)
        states = yback[-1, :]
    return v*4000/2.2, delta*180/np.pi


def follow_line(frame, ultraschall_sub, vel=0.6):
    #verolgen der linie hier einfuegen
    if ultraschall_sub.range < 20:
        global linie_folgen, fahrzeug_folgen, x_0, x_1, y_0, y_1, alpha_car, timer_always
        linie_folgen = False
        fahrzeug_folgen = True
        x_0 = 326
        y_0 = 396
        x_1 = 400
        y_1 = 398
        alpha_car = 0
        timer_always = 0
    return vel * 4000 / 2.2, 0


def follow_car(frame, tracker):
    global x_0, y_0, x_1, y_1, alpha_car, x_car, y_car, timer_always, ueberholen, fahrzeug_folgen,angle_03,states,u_f,u_b
    x_0_old = x_0
    y_0_old = y_0
    x_1_old = x_1
    y_1_old = y_1
    alpha_old = alpha_car
    x_0, y_0, x_1, y_1, can_see = tracker.get_red_pos(frame, x_0, y_0, x_1, y_1)
    x_0 = tiefpass(x_0, x_0_old)
    y_0 = tiefpass(y_0, y_0_old)
    x_1 = tiefpass(x_1, x_1_old)
    y_1 = tiefpass(y_1, y_1_old)

    drehung = -np.arctan2(y_1 - y_0, x_1 - x_0) * 2.9
    drehung = drehung * 180 / np.pi
    # print("drehung", float(drehung))
    fov = 62.2
    f = 592.61
    hoehe_cam = 220
    de = 73
    dp = x_1 - x_0
    alpha_car = -((x_1 + x_0) / 2 - 768 / 2) / (768 / 2) * fov / 2 / 180 * np.pi * 1.08
    alpha_car = tiefpass(alpha_car, alpha_old, 0.8)
    distance = de * f / dp * (1 + abs(alpha_car) / 4 * 1.1) * (1 - abs(drehung / 100) / 2)
    distance = np.sqrt(distance ** 2 - hoehe_cam ** 2)
    x_car = np.cos(alpha_car) * distance
    y_car = np.sin(alpha_car) * distance
    accell_in, steering = controller_verfolgen(x_car, y_car, alpha_car)

    if timer_always > 1:
        ueberholen = True
        fahrzeug_folgen = False
        angle_03=0
        x_f = x_car / 1000
        y_f = y_car /1000
        betha_f = 0.0
        psi_f = alpha_car
        phi_f = 0.0
        v_f = accell_in/4000*2.2
        delta_f = 0.0

        x_b = 0.0
        y_b = 0.0
        betha_b = 0.0
        psi_b = 0.0
        phi_b = 0.0

        v_b = accell_in/4000*2.2
        delta_b = 0.0

        u_f = np.array([v_f, delta_f])

        u_b = np.array([v_b, delta_b])
        states = np.array([x_b, y_b, betha_b, psi_b, phi_b, x_f, y_f, betha_f, psi_f, phi_f])

    return accell_in, steering


def callback(image_sub, imu_03_sub, ultraschall_sub, car_f_sub, imu_10_sub):
    #print("huhu")
    frame = transform_to_opencv(image_sub)
    global ueberholen, linie_folgen, fahrzeug_folgen, timer_always,inverse_model,tracker

    if ueberholen:
        print("Ueberholen")
        accell_in, steering = overtake(imu_03_sub, car_f_sub, imu_10_sub,inverse_model)
    elif linie_folgen:
        print("following_line")
        accell_in, steering = follow_line(frame, ultraschall_sub)
    elif fahrzeug_folgen:
        print("verfolgen")
        accell_in, steering = follow_car(frame,tracker)
    timer_always = timer_always + 1 / 20.0

    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = accell_in  # aktuell in tick rate(+- 3900)
    message.point.y = 2  # not used
    message.point.z = steering  # in grad(max +-20)
    # rospy.loginfo(message)
    pub.publish(message)
    rate.sleep()

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    global tracker,inverse_model
    tracker = tracking_red_dots(576, 768)
    inverse_model = imcr.invModelControl(speed, 0.4, "sShape")
    image_sub = message_filters.Subscriber('/raspicam_node/image', Image)
    ultraschall_sub = message_filters.Subscriber('/distance_sensor_03', Range)
    imu_03_sub = message_filters.Subscriber('/IMU_03', Imu)
    imu_10_sub = message_filters.Subscriber('/IMU_10', Imu)
    car_f_sub = message_filters.Subscriber('/car_input_10', PointStamped)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, imu_03_sub, ultraschall_sub, car_f_sub, imu_10_sub],
                                                     100, 5)
    ts.registerCallback(callback)
    # rospy.init_node('listener', anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    listener()
