import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        self.aruco_subscriber = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()

        # camera parameters
        self.camera_matrix = np.array([[800, 0, 320],
                                       [0, 800, 240],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # --- ArUco Dictionary ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.marker_size = 4 # cm

    def image_callback(self, msg):
        # read image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print(type(frame), frame.shape)

        # convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            # draw green boxes around detected markers
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
            # rvecs: rotation vectors
            # tvecs: translation vectors (gives 3D position of the marker relative to the camera)
            
            for i in range(len(ids)):
                # Draw axis
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 2)            # Display distance (norm of tvec)
                
                distance = np.linalg.norm(tvecs[i][0])
                cv2.putText(frame, f"id: {ids[i]}",
                            (int(corners[i][0][0][0]) - 40, int(corners[i][0][0][1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"x: {tvecs[i][0][0]: .2f}",
                            (int(corners[i][0][0][0]) - 40, int(corners[i][0][0][1]) + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"y: {tvecs[i][0][1]: .2f}",
                            (int(corners[i][0][0][0]) - 40, int(corners[i][0][0][1]) + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"z: {tvecs[i][0][2]: .2f}",
                            (int(corners[i][0][0][0]) - 40, int(corners[i][0][0][1]) + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  
                # cv2.putText(frame, f"Distance: {distance:.2f} cm",
                #             (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
