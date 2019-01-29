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
# from invModelControlforROS import invModelControl
rospy.init_node('publisher', anonymous=True)
pub = rospy.Publisher('car_input_10', PointStamped, queue_size=1)
rate = rospy.Rate(20)  # Frequenz der Anwendung
tracker = None
inverse_model = None
ueberholen = False
linie_folgen = False
fahrzeug_folgen = True
x_0 = 326
y_0 = 396
x_1 = 400
y_1 = 398
alpha_car = 0
x_car = 0
y_car = 0
angle_03 = 0
accel_in_f_ma = 0
speed = 1.1 * 0.8
timer_always = 0
old_time = 0
old_time_for_simulation = 0
# simulierungs variablen
states = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # np.array([x_b, y_b, betha_b, psi_b, phi_b, x_f, y_f, betha_f, psi_f, phi_f])

u_f = np.array([0, 0])

u_b = np.array([0, 0])
t_range = 1 / 20.0


def transform_to_opencv(image):
    bridge = CvBridge()
    try:
        #image = bridge.imgmsg_to_cv2(image, "bgr8")
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

i_anteil=0
def controller_verfolgen(x, y, alpha):
    global i_anteil
    x = x - 350
    a = -y / x ** 3 + np.tan(alpha) / x ** 2
    b = y / x ** 2 - a * x
    y_pos = cubic_function(x / 2, a, b, 0, 0)
    steering = np.arctan2(y_pos, x / 2)
    steering = maxValue(steering * 180 / 3.14159, 29)
    i_anteil=i_anteil+x / 2 - 150 / 2
    #print("i anteili",i_anteil)
    i_anteil=maxValue(i_anteil,2000)
    if i_anteil<0:
        i_anteil=0
    v_wanted = 8 * (x / 2 - 50 / 2)+i_anteil*0.15
    accell_in = maxValue( v_wanted, 4000)
    return accell_in, steering

d_anteil=0
alter_error=0
def overtake(imu_03_sub, inverse_model):
    global angle_03, timer_always, speed, states, u_b, u_f, linie_folgen, ueberholen,old_time_for_simulation,d_anteil,alter_error
    scaling_imu_angle = 17063036.0 / 4.0 / 360.0 * 5 / 45.0*180.0/140.0
    angle_03 = angle_03 + imu_03_sub.angular_velocity.z
    # v = speed
    timer_always=float(timer_always)
    initialdelaytime = 0
    middledelaytime = 0
    if timer_always < initialdelaytime:
        v = speed
        inverse_model.trajectory.setSpecifics(Vsoll=(speed*1.1) , Vstart=speed )
        delta = 0
    else:
        if timer_always > inverse_model.trajectory.specifics.T + initialdelaytime + middledelaytime:
            print("x_pos",states[0]-0.36, states[5])
            if states[0]-0.36 > states[5]:
                if not inverse_model.T0:
                    print("init")
                    inverse_model.T0 = timer_always
                    # inverse_model.trajectory.setSpecifics([v,-inverse_model.trajectory.specifics.W])
                    inverse_model.trajectory.specifics.setVstart(inverse_model.trajectory.specifics.Vsoll)

                #print("in seconds:", timer_always - inverse_model.T0)
                v, delta, psi = inverse_model.carInput(timer_always - inverse_model.T0)
                delta = -delta
                psi = -psi
                #print("time wieder einschehren",inverse_model.trajectory.specifics.T)
                if timer_always > 1.5 * (inverse_model.trajectory.specifics.T) + inverse_model.T0:
                    v = inverse_model.trajectory.specifics.Vsoll
                    inverse_model.T0=None
                    linie_folgen = True
                    ueberholen = False
                    inverse_model.T0 = None

            else:
                v = inverse_model.trajectory.specifics.Vsoll
                delta = 0
                psi = 0
        # elif timer_always > inverse_model.trajectory.specifics.T + initialdelaytime and timer_always < inverse_model.trajectory.specifics.T + initialdelaytime + middledelaytime:
        #     delta = 0
        #     psi = 0
        else:
            #print("timer always ueberholen",timer_always,inverse_model.trajectory.specifics.T)
            v, delta, psi = inverse_model.carInput(timer_always - initialdelaytime)
        psi = -psi * 180 / 3.14159
        delta = -delta*1.1
        error = psi - angle_03 / scaling_imu_angle

        d_anteil=tiefpass(error-alter_error,d_anteil,0.99)
        print("d_anteil:",d_anteil)
        #inverse_correction = inverse_model.trajectoryControler(error, 1.5 * 20) / 180 * 3.14159
        inverse_correction = 0.5*error * 1 / 180.0 * 3.14159+0.7*d_anteil/180*3.14159
        alter_error = error
        funneltime = timer_always
        if inverse_model.T0:
            funneltime = funneltime - inverse_model.T0
        deltamax = 10
        #delta = inverse_model.funnelControler(-error,funneltime,deltamax) #delta +
        delta = delta + inverse_correction
        # print(delta, inverse_correction)
        # delta=np.pi/4
        u_b[0] = v
        # u_b[1] = delta - inverse_model.funnelControler(-error,funneltime,deltamax)
        u_b[1] = delta - inverse_correction
        # print(u_b[1])
        # print(t_range)
        # print(type(t_range))
        # print(states)
        # print(u_b)
        # print(u_f)
        current = rospy.get_time()  # Time.now()
        t_range=current-old_time_for_simulation
        old_time_for_simulation = rospy.get_time()
        #print("T_RANGE:",t_range)
        states=np.array(states, dtype='float64')
        yback = inverse_model.simulateModel(states, t_range, model="discrete", ub=u_b, uf=u_f)

        #print("simulationszeit:", rospy.get_time() - current)
        states = yback[-1, :]

    return v * 4000 / 2.2, delta * 180 / np.pi, angle_03 / scaling_imu_angle


def follow_line(frame, vel=0.6):
    # verolgen der linie hier einfuegen
    if 10 < 45 * 1.2*0:  # 45 cm
        global linie_folgen, fahrzeug_folgen, x_0, x_1, y_0, y_1, alpha_car, timer_always
        linie_folgen = False
        fahrzeug_folgen = True
        x_0 = 326 + 30
        y_0 = 396 - 20
        x_1 = 400 + 30
        y_1 = 398 - 20
        alpha_car = 0
        timer_always = 0
        # return 0, 0
    return vel * 4000 / 2.2*0, 0


def follow_car(frame, tracker):
    global x_0, y_0, x_1, y_1, alpha_car, x_car, y_car, timer_always, ueberholen, fahrzeug_folgen, angle_03, states, inverse_model, u_f, u_b, accel_in_f_ma, speed,old_time_for_simulation
    x_0_old = x_0
    y_0_old = y_0
    x_1_old = x_1
    y_1_old = y_1
    alpha_old = alpha_car
    x_0, y_0, x_1, y_1, _ = tracker.get_red_pos(frame, x_0, y_0, x_1, y_1)
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
    # print(alpha_car)
    accell_in, steering = controller_verfolgen(x_car, y_car, alpha_car)

    accel_in_f_ma = tiefpass(accell_in, accel_in_f_ma, 0.95)

    if timer_always > 4:
        ueberholen = True
        fahrzeug_folgen = False
        angle_03 = 0
        x_f = x_car / 1000
        y_f = y_car / 1000
        betha_f = 0.0
        psi_f = alpha_car
        phi_f = 0.0
        v_f = accel_in_f_ma / 4000 * 2.2  # moving average
        delta_f = 0.0
        print("speed vorderauto",accel_in_f_ma)
        x_b = 0.0
        y_b = 0.0
        betha_b = 0.0
        psi_b = 0.0
        phi_b = 0.0

        v_b = accell_in / 4000 * 2.2
        delta_b = steering

        u_f = np.array([v_f, delta_f])

        u_b = np.array([v_b, delta_b])
        states = np.array([x_b, y_b, betha_b, psi_b, phi_b, x_f, y_f, betha_f, psi_f, phi_f])
        # inverse_model.trajectory.specifics.setVstart(accell_in / 4000 * 2.2)
        # inverse_model.trajectory.updateVsoll((accell_in + 700) / 4000 * 2.2)
        inverse_model.trajectory.setSpecifics(Vsoll=(accell_in + 700) / 4000 * 2.2, Vstart=accell_in / 4000 * 2.2)
        print("gross T",inverse_model.trajectory.specifics.T)
        speed = accell_in
        old_time_for_simulation=rospy.get_time()
        timer_always = 0

    return accell_in, steering


def callback(image_sub, imu_03_sub):#ultraschall_sub):
    #print("huhu")
    frame = transform_to_opencv(image_sub)
    global ueberholen, linie_folgen, fahrzeug_folgen, timer_always, inverse_model, tracker, speed, old_time
    imu_angle = 0
    if ueberholen:
        print("Ueberholen")
        accell_in, steering, imu_angle = overtake(imu_03_sub, inverse_model)
    elif fahrzeug_folgen:
        print("verfolgen")
        accell_in, steering = follow_car(frame, tracker)
    elif linie_folgen:
        print("following_line")
        accell_in, steering = follow_line(frame, vel=speed)
    # print("time since last call",(rospy.get_time()-old_time))
    # timer_always = timer_always + 1 / 20.0
    timer_always = timer_always + (rospy.get_time() - old_time)
    old_time = rospy.get_time()
    #print("time Always",timer_always)
    # print(accell_in)
    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = accell_in*1.5  # aktuell in tick rate(+- 3900)
    message.point.y = imu_angle  # not used
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
    global tracker, inverse_model, old_time
    tracker = tracking_red_dots(576, 768)
    inverse_model = imcr.invModelControl(speed, 0.4, "quadS")
    image_sub = message_filters.Subscriber('/raspicam_node/image', Image)
    #ultraschall_sub = message_filters.Subscriber('/distance_sensor_03', Range)
    imu_03_sub = message_filters.Subscriber('/IMU_10', Imu)
    old_time = rospy.get_time()
    # imu_10_sub = message_filters.Subscriber('/IMU_10', Imu)
    # car_f_sub = message_filters.Subscriber('/car_input_10', PointStamped)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, imu_03_sub],
                                                     10, 1)
    ts.registerCallback(callback)
    # rospy.init_node('listener', anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    listener()
