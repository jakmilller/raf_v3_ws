#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import tf
import tf.transformations as tf_trans
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from math import sqrt, inf, degrees, radians
from rs_ros import RealSenseROS
from pixel_selector import PixelSelector
import raf_utils
from cv_bridge import CvBridge
import math
from scipy.spatial.transform import Rotation

# Initialize CvBridge
bridge = CvBridge()

class CamDetection:
    def __init__(self):
        self.pixel_selector = PixelSelector()
        self.tf_utils = raf_utils.TFUtils()
    
    def grasping_pretzels(self, color_image, depth_image, camera_info, isOpenCv=False):
        if isOpenCv:
            (center_x, center_y, pretzel_angle) = self.detectPretzel(color_image)
            major_axis = 0
        else:
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            major_axis = 0

        print(f"Center x {center_x}, Center y {center_y}")

    
        # get 3D point from depth image
        validity, point = raf_utils.pixel2World(camera_info, center_x, center_y, depth_image)
        if not validity:
            print("Invalid point")
            return
        print(f"3D point: {point}")

        food_transform = np.eye(4)
        food_transform[:3,3] = point.reshape(1,3)
        food_base = self.tf_utils.getTransformationFromTF("base_link", "camera_link") @ food_transform
        base_coords = food_base[:3,3] 
        base_coords[1] += 0.05
        rotataion_matrix = food_base[:3,:3]

        

        euler_angles = Rotation.from_matrix(rotataion_matrix).as_euler('xyz', degrees=True)
        euler_angles[2] = -euler_angles[2]
        print(f"Coordinates: x: {base_coords[0]} y: {base_coords[1]} z: {base_coords[2]} roll: {euler_angles[0]} pitch: {euler_angles[1]} yaw: {pretzel_angle}")

        if major_axis < np.pi/2:
            major_axis = major_axis + np.pi/2

    
    def pretzel_orientation(self, cv_image):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

                # Apply thresholding to create a binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through each contour
        for contour in contours:
            if len(contour) > 0:
                # Calculate the image moments
                moments = cv2.moments(contour)

                # Compute the centroid of the object
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = 0, 0

                # Calculate orientation using central moments
                mu20 = moments['mu20']
                mu02 = moments['mu02']
                mu11 = moments['mu11']
                
                # Compute orientation angle
                angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
                angle_degrees = np.degrees(angle)
                # Keep the angle between 0 to 180 degrees
                if angle_degrees < 0:
                    angle_degrees += 180

                print(f"Orientation: {angle_degrees:.2f} degrees")

    def calculate_pretzel_orientation(self, largest_brown_contour, image):
        """
        Calculate the orientation of the pretzel from its contour.
        Returns: angle of rotation in radians
        """
        # Fit an ellipse to the contour (this method requires at least 5 points)
        if len(largest_brown_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_brown_contour)
            center, axes, angle = ellipse
            cx, cy = int(center[0]), int(center[1])  # Center of the ellipse
            major_axis_length = max(axes) / 2  # Half of the major axis length
            minor_axis_length = min(axes) / 2  # Half of the minor axis length

            # Adjust the angle by adding 90 degrees to align with the robot's end-effector
            # if angle > 80 and angle < 100:
            #     angle -= 90
            # elif angle > 0 and angle < 30:
            #     angle = 90

            # The angle is in degrees, convert it to radians
            angle_rad = math.radians(angle)

            # Draw the major axis (in the direction of the pretzel's orientation)
            major_axis_x = int(cx + major_axis_length * math.cos(angle_rad))
            major_axis_y = int(cy + major_axis_length * math.sin(angle_rad))
            cv2.line(image, (cx, cy), (major_axis_x, major_axis_y), (0, 255, 0), 2)  # Green line for major axis

            # Draw the minor axis (perpendicular to the major axis)
            minor_axis_x = int(cx - minor_axis_length * math.sin(angle_rad))
            minor_axis_y = int(cy + minor_axis_length * math.cos(angle_rad))
            cv2.line(image, (cx, cy), (minor_axis_x, minor_axis_y), (255, 0, 0), 2)  # Blue line for minor axis

            # Draw the center of the ellipse
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)  # Red dot at the center

            return angle_rad, center, angle
        else:
            return None, None, None
    
    def detectPretzel(self, cv_image):
        # Convert the image from BGR to HSV color space
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define the range of brown color in HSV (you might need to fine-tune this)
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])

        # Threshold the HSV image to get only brown colors
        mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

        # Find contours of the brown areas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour which will likely be the pretzel
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Filter small areas
                # Get the center of the contour (centroid)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Draw the contour and the center point on the image (optional for debugging)
                    cv2.drawContours(cv_image, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(cv_image, (cX, cY), 7, (255, 0, 0), -1)
                    cv2.putText(cv_image, "Pretzel Center", (cX - 20, cY - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    angle_rad, pretzel_center, pretzel_angle = self.calculate_pretzel_orientation(largest_contour, cv_image)
                    self.pretzel_orientation(cv_image)

                    if pretzel_center is not None:
                        # Calculate orientation of the pretzel and adjust the robot's end-effector accordingly
                        rospy.loginfo(f"Orientation of the pretzel: {degrees(angle_rad)} degrees")
                        

                    #rospy.loginfo(f"Detected pretzel center at pixel: ({cX}, {cY})")
                    # Display the image
                    cv2.imshow("Pretzel Detection", cv_image)
                    cv2.waitKey(1)  # Wait for a short period to allow the image to be displayed
                    
                    # Return the coordinates of the center pixel
                    return cX, cY, pretzel_angle
        return None, None, None
        

def main():
    rospy.init_node('cam_detection', anonymous=True)
    cd = CamDetection()
    camera = RealSenseROS()
    camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()
    cd.grasping_pretzels(camera_color_data, camera_depth_data, camera_info_data, isOpenCv=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:    
        cv2.destroyAllWindows()


    

if __name__ == '__main__':
    main()
