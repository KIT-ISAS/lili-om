#!/usr/bin/python

import tf
import rospy
import math
from sensor_msgs.msg import Imu

class imu_rescaller:
  def __init__(self):
    self.unit_acc = 9.8
    self.orientation_avg_num = 3

    pub_topic_name = "/imu/data"
    sub_topic_name = "/livox/imu"

    self.pub = rospy.Publisher(pub_topic_name, Imu, queue_size = 200)
    rospy.Subscriber(sub_topic_name, Imu, self.callback)

    self.orientation_avg_count = 0
    self.imu_first_msg = Imu()
    
    self.imu_first_msg.linear_acceleration.x = 0
    self.imu_first_msg.linear_acceleration.y = 0
    self.imu_first_msg.linear_acceleration.z = 0

  def callback(self, imu):
    msg = Imu()
    msg = imu
    
    msg.linear_acceleration.x = imu.linear_acceleration.x * self.unit_acc
    msg.linear_acceleration.y = imu.linear_acceleration.y * self.unit_acc
    msg.linear_acceleration.z = imu.linear_acceleration.z * self.unit_acc

    if self.orientation_avg_count < self.orientation_avg_num:
      self.imu_first_msg.linear_acceleration.x += msg.linear_acceleration.x
      self.imu_first_msg.linear_acceleration.y += msg.linear_acceleration.y
      self.imu_first_msg.linear_acceleration.z += msg.linear_acceleration.z

      self.orientation_avg_count += 1
      
      if self.orientation_avg_count == self.orientation_avg_num:
        self.imu_first_msg.linear_acceleration.x /= self.orientation_avg_num
        self.imu_first_msg.linear_acceleration.y /= self.orientation_avg_num
        self.imu_first_msg.linear_acceleration.z /= self.orientation_avg_num
        
        ax = self.imu_first_msg.linear_acceleration.x
        ay = self.imu_first_msg.linear_acceleration.y
        az = self.imu_first_msg.linear_acceleration.z

        r = math.atan2(ay, az)
        p = math.atan2(-ax, math.sqrt(ay**2 + az**2))
        y = 0

        q = tf.transformations.quaternion_from_euler(r, p, y)
        self.imu_first_msg.orientation.x = q[0]
        self.imu_first_msg.orientation.y = q[1]
        self.imu_first_msg.orientation.z = q[2]
        self.imu_first_msg.orientation.w = q[3]
        
    else: 
      msg.orientation = self.imu_first_msg.orientation
      self.pub.publish(msg)

def main():
    rospy.init_node("rescale_imu")
    rescaller = imu_rescaller()
    rospy.spin()

if __name__ == "__main__":
    main()
