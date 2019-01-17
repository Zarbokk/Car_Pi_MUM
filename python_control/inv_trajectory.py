#!/usr/bin/env python
# license removed for brevity
import rospy
from car_scripts import invModelControlROS as imcr
import numpy as np
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

pub = rospy.Publisher('car_input_03', PointStamped, queue_size=1)
plot_data = rospy.Publisher("plot_data", Odometry, queue_size=1)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(50)  # Frequenz der Anwendung
start_time = 0.0
angle = 0.0
linear_vel_x = 0.0
linear_vel_y = 0.0
geschwindigkeit = 1.1 * 1.2
x_f = 650.0 / 1000
y_f = 0.0
betha_f = 0.0
psi_f = 0.0
phi_f = 0.0
v_f = 0.0
delta_f = 0.0

x_b = 0.0
y_b = 0.0
betha_b = 0.0
psi_b = 0.0
phi_b = 0.0
v_b = 0.0
delta_b = 0.0

v = 0

states = np.array([x_b, y_b, betha_b, psi_b, phi_b, x_f, y_f, betha_f, psi_f, phi_f])

u_f = np.array([v_f, delta_f])

u_b = np.array([v_b, delta_b])
t_range = 1 / 50.0


def talker(Imu_data, inverse_model):
    global angle
    global start_time, linear_vel_x, linear_vel_y
    global states, u_f, u_b, t_range, v
    y = 0
    error = 0
    psi = 0
    scaling_imu_angle = 17063036.0 / 4.0 / 360.0 * 2
    angle = angle + Imu_data.angular_velocity.z
    linear_vel_x = linear_vel_x + Imu_data.linear_acceleration.x
    linear_vel_y = linear_vel_y + Imu_data.linear_acceleration.y
    real_angle = angle

    endtime = inverse_model.trajectory.specifics.T
    # print(start_time, endtime)

    initialdelaytime = 1
    middledelaytime = 1

    if start_time < initialdelaytime:
        v = geschwindigkeit
        delta = 0
    else:
        if start_time > inverse_model.trajectory.specifics.T + initialdelaytime + middledelaytime:
            if states[0] > states[5]:
                if not inverse_model.T0:
                    print("init")
                    inverse_model.T0 = start_time
                    # inverse_model.trajectory.setSpecifics([v,-inverse_model.trajectory.specifics.W])
                    inverse_model.trajectory.updateVsoll(v)
                print("in second")
                print(start_time - inverse_model.T0)
                v, delta, psi = inverse_model.carInput(start_time - inverse_model.T0)
                delta = -delta
                psi = -psi
                print(delta)
                if start_time > 1.5 * (inverse_model.trajectory.specifics.T + inverse_model.T0):
                    v = 0
                    states[0] = 0
                    states[1] = 0

            else:
                delta = 0
                psi = 0
        elif start_time > inverse_model.trajectory.specifics.T + initialdelaytime and start_time < inverse_model.trajectory.specifics.T + initialdelaytime + middledelaytime:
            delta = 0
            psi = 0
        else:
            v, delta, psi = inverse_model.carInput(start_time - initialdelaytime)
        psi = -psi * 180 / 3.14159
        delta = -delta
        error = psi - real_angle / scaling_imu_angle
        # inverse_correction = inverse_model.trajectoryControler(error, 1.5 * 20) / 180 * 3.14159
        inverse_correction = error * 1 / 180.0 * 3.14159
        delta = delta + inverse_correction
        # print(delta, inverse_correction)
        # delta=np.pi/4
        u_b[0] = v
        u_b[1] = delta - inverse_correction
        # print(u_b[1])
        yback = inverse_model.simulateModel(states, t_range, model="discrete", ub=u_b, uf=u_f)
        states = yback[-1, :]
        # print(delta)
        # print(states)
    # v = 2.2
    # delta = 0

    # delta = 0
    # v = 0
    # delta = 0

    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = v / 2.2 * 4095  # aktuell in tick rate(+- 3900)
    message.point.y = real_angle / scaling_imu_angle  # (np.sqrt(linear_vel_y*linear_vel_y+linear_vel_x*linear_vel_x)-400*100*start_time)/3000000*2.2*4*2  # not used
    message.point.z = delta * 180 / 3.14159  # in grad(max +-20)
    # rospy.loginfo(message)
    pub.publish(message)

    ##PLOTTTING###

    message_2 = Odometry()
    message_2.header.stamp = rospy.Time.now()
    message_2.pose.pose.orientation.x = states[0]
    message_2.pose.pose.orientation.y = states[1]
    message_2.pose.pose.orientation.z = states[2]
    message_2.pose.pose.orientation.w = states[3]
    message_2.pose.pose.position.x = states[4]
    message_2.pose.pose.position.y = states[5]
    message_2.pose.pose.position.z = states[6]
    message_2.twist.twist.linear.x = states[7]
    message_2.twist.twist.linear.y = states[8]
    message_2.twist.twist.linear.z = states[9]
    plot_data.publish(message_2)

    rate.sleep()
    start_time = start_time + 0.02


def subscribe():
    inverse_model = imcr.invModelControl(geschwindigkeit, 0.4, "sShape")

    while not rospy.is_shutdown():
        rospy.Subscriber('IMU_03', Imu, talker, inverse_model)
        rospy.spin()


# +90
if __name__ == '__main__':
    try:
        subscribe()
    except rospy.ROSInterruptException:
        pass
