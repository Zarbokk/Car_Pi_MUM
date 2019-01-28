# !/usr/bin/env python

if __name__ == '__main__':
    from cv_bridge import CvBridge, CvBridgeError
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import Image, CompressedImage
    import rospy
    rospy.init_node('publisher', anonymous=True)
    pub = rospy.Publisher('car_input_03', PointStamped, queue_size=1)
    rate = rospy.Rate(20)  # Frequenz der Anwendung
import numpy as np
import cv2
import matplotlib as mpl


def get_warped(this_img, height_pct=.4, return_m_inv=False):
    """Get Warping."""
    this_img_size = np.float32([(this_img.shape[1], this_img.shape[0])])
    img_size = (this_img.shape[1], this_img.shape[0])
    bot_width = .75  # percentage of bottom trapezoidal height
    mid_width = .17  # percentage of mid trapezoidal height
    height_pct = 1 - height_pct  # percentage of trapezoidal height
    bottom_trim = .99
    # percentage from top to bottom avoiding the hood of the car

    src = np.float32([
        (0.5 - mid_width / 2, height_pct),
        (0.5 + mid_width / 2, height_pct),
        (0.5 + bot_width / 2, bottom_trim),
        (0.5 - bot_width / 2, bottom_trim)])
    src = src * this_img_size
    offset = img_size[0] * 0.25
    dst = np.float32([[
        offset, 0],
        [img_size[0] - offset, 0],
        [img_size[0] - offset, img_size[1]],
        [offset, img_size[1]]])

    warped = cv2.warpPerspective(
        this_img, cv2.getPerspectiveTransform(src, dst), img_size)
    if return_m_inv:
        minv = cv2.getPerspectiveTransform(dst, src)
        return warped, src, dst, minv
    else:
        return warped, src, dst


def plot_warping(ax, src, dst, this_img, warped):
    """Helper for plotting the warping, not relevant for car control."""
    pol = mpl.patches.Polygon(src, fill=False)
    ax[0].imshow(this_img)
    ax[0].scatter(*zip(*src), marker="s", s=30, c="r")
    ax[0].add_patch(pol)
    ax[1].imshow(warped)
    ax[1].scatter(*zip(*dst), marker="s", s=30, c="r")
    for axx in ax:
        axx.set_xticks([])
        axx.set_yticks([])

def plot_result(ax, warped, out_img, fit):
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    fitx = np.polyval(fit, ploty)
    ax.imshow(np.max(out_img, axis=2) + warped)
    ax.plot(fitx, ploty, color='yellow', linestyle="dashed", linewidth=5)
    ax.set_title(np.round(np.mean(np.polyval(np.polyder(fit), ploty[len(ploty)//4*3:])),3)/1.97*29*(-1))



def calc_line_fits(img, x_base=None, nwindows=15, margin=20, minpix=30,
                   degree=3, get_image=False, debug=False, previous_fit=None):
    """Calculate Lane Line by histogram approach and interpolating polynomial.

    Parameters
    ----------
    img : array
        Image file, must be array of gray values -> shape:(N,N,1)
    x_base : int, optional
        Where to start, e.g. where to center first window
    nwindows : int, optional
        Choose the number of sliding windows
    margin : int, optional
        Set the width of the windows +/- margin
    minpix : int, optional
        Set minimum number of pixels found to recenter window
    degree : int, optional
        Degree of fitting Polynomial
    get_image : bool, optional
        Description

    Returns
    -------
    fit : np.array
        Coefficients of polynomial, use np.polyval(p, x)
    out_img : np.array
        Image with plotted rectangles
    """
    minpix = minpix * nwindows

    if x_base:
        pass
    else:
        # Take histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0] // 3:, :], axis=0)
        # Find the peak of the histogram
        midpoint = np.int(histogram.shape[0] / 2)
        cut_nenner = 3
        x_base = np.argmax(histogram[
            midpoint * (cut_nenner - 1) / cut_nenner:
            midpoint * (cut_nenner + 1) / cut_nenner])
    # Create an output image to draw on and  visualize the result
    if get_image:
        out_img = np.dstack((img, img, img)) * 255
        # histogram = np.sum(img[img.shape[0] // 3:, :], axis=0)
        # # Find the peak of the histogram
        # midpoint = np.int(histogram.shape[0] / 2)
        # cut_nenner = 3
        # cv2.rectangle(
        #     out_img,
        #     (midpoint * (cut_nenner - 1) / cut_nenner, img.shape[1]-margin),
        #     (midpoint * (cut_nenner + 1) / cut_nenner, img.shape[1]),
        #     (0,0,0), thickness=3
        # )
        cv2.circle(out_img, (x_base, img.shape[0]), 30, (0, 0, 0), thickness=2)
    if previous_fit is not None:
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        move_by = np.zeros(len(previous_fit))
        move_by[-1] = 20  # 20 px linear moving
        move_by[-2] = .5
        upper_bound = np.polyval(previous_fit + [0, 0, .5, 20], ploty)
        lower_bound = np.polyval(previous_fit - [0, 0, .5, 20], ploty)
        for (i, j) in np.ndindex(img.shape):
            if j > upper_bound[i] or j < lower_bound[i]:
                img[i, j] = 50
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    x_current = x_base
    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        if window == 0:
            win_x_low -= 2 * margin
            win_x_high += 2 * margin

        # Draw the windows on the visualization image
        if get_image:
            cv2.rectangle(
                out_img, (win_x_low, win_y_low), (win_x_high, win_y_high),
                (0, 0, 0), thickness=2)
        # Identify the nonzero pixels in x and y within the window
        good_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        # If you found > minpix pixels,
        # recenter next window on their mean position
        if len(good_inds) > minpix:
            if debug:
                print("step {}: {} / {}".format(
                    window, len(good_inds),
                    (win_x_high - win_x_low) * (win_y_high - win_y_low)
                ))
            # Append these indices to the lists
            lane_inds.append(good_inds)
            if np.sum(img[win_y_low:win_y_high, win_x_low:win_x_high]):
                x_current = win_x_low + np.argmax(np.sum(
                    img[win_y_low:win_y_high, win_x_low:win_x_high], axis=0))
            else:
                pass

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    # Fit a second order polynomial to each
    fit = np.polyfit(y, x, degree)

    if get_image:
        return fit, out_img
    else:
        return fit


def transform_to_opencv(image):
    """Transform Image Message to Opencv Image."""
    bridge = CvBridge()
    try:
        image = bridge.compressed_imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)
    return image

angle1 = 0
angle2 = 0
angle3 = 0


def callback(image_sub):
    """Callback for Processing the stuff."""
    # use last angles
    global angle1, angle2, angle3
    angle1 = angle2
    angle2 = angle3

    frame = transform_to_opencv(image_sub)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # np.save('images/{}'.format(rospy.Time.now()), frame)
    warped, src, dst = get_warped(frame)
    steering = 0
    try:
        fit = calc_line_fits(
            warped, nwindows=25, x_base=warped.shape[1]//2-20, minpix=10,
            get_image=False, degree=2)
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        # left_fitx = np.polyval(left_fit, ploty)
        # np.mean(1 / (100 + ploty) * 100 * np.polyval(
        #     np.polyder(left_fit), ploty))
        steering = np.round(np.mean(np.polyval(np.polyder(fit), ploty[len(ploty)//4*3:])),3)/1.97*29*(-1)
        steering = steering if abs(steering) <= 29 else np.sign(steering) * 29
    except Exception as e:
        print('error:\t{}'.format(e))

    angle3 = np.sign(angle3) * min(29, 2 * abs(angle3) / 2.9 * 29)
    speed = 0
    # speed = 2000 * np.exp(-abs(steering) / 21)
    # speed = speed if speed > 500 else 500

    print("{:.3f}\t({:.3f}\t{:.3f}\t{:.3f}),\t{}".format(
        steering, angle1, angle2, angle3, speed))

    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = speed
    message.point.y = 0
    message.point.z = steering  # in grad(max +-20)
    pub.publish(message)
    rate.sleep()


def listener():
    global angle1, angle2, angle3
    rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
