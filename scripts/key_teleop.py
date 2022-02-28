#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 PAL Robotics SL.
# Released under the BSD License.
#
# Authors:
#   * Siegfried-A. Gevatter

import curses

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool

class TextWindow():

    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=10):
        self._screen = stdscr
        self._screen.nodelay(True)
        curses.curs_set(0)

        self._num_lines = lines

    def read_key(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def write_line(self, lineno, message):
        if lineno < 0 or lineno >= self._num_lines:
            raise ValueError('lineno out of bounds')
        height, width = self._screen.getmaxyx()
        y = int((height / self._num_lines) * lineno)
        x = 10
        for text in message.split('\n'):
            text = text.ljust(width)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()



class SimpleKeyTeleop():
    def __init__(self, interface):
        self._interface = interface
        self._pub_cmd = rospy.Publisher('/key_teleop/vel_cmd_body_frame', Twist)
        self._pub_cmd_attack = rospy.Publisher('/key_teleop/attack_on_off', Bool)

        self._hz = rospy.get_param('~hz', 10)

        self._forward_rate = rospy.get_param('~forward_rate', 5.0)
        self._backward_rate = rospy.get_param('~backward_rate', 5.0)
        self._side_rate = rospy.get_param('~side_move_rate', 5.0)
        self._elevation_rate = rospy.get_param('~elevation_rate', 5.0)

        self._rotation_rate = rospy.get_param('~rotation_rate', 5.0)
        self._last_pressed = {}
        self._angular = (0, 0, 0)
        self._linear = (0, 0, 0)
        self.m = -1

        self._image_attack_bool = Bool()
        self._image_attack_bool.data=True

    movement_bindings = {
        curses.KEY_UP:    (+1,  0,  0,  0,  0,  0,  0),  # +x forward
        curses.KEY_DOWN:  (-1,  0,  0,  0,  0,  0,  0),  # -x backward
        curses.KEY_LEFT:  ( 0, -1,  0,  0,  0,  0,  0),  # +y 
        curses.KEY_RIGHT: ( 0, +1,  0,  0,  0,  0,  0),  # -y
        ord('w'):         ( 0,  0, +1,  0,  0,  0,  0),  # +z
        ord('s'):         ( 0,  0, -1,  0,  0,  0,  0),  # -z
        ord('a'):         ( 0,  0,  0,  0,  0, -1,  0),  # +yaw
        ord('d'):         ( 0,  0,  0,  0,  0, +1,  0),  # -yaw
        ord('e'):         ( 0,  0,  0,  1,  0,  0,  0),  # take off
        ord('q'):         ( 0,  0,  0, -1,  0,  0,  0),  # landing
        ord('c'):         ( 0,  0,  0,  0,  1,  0,  0),  # engage the controller
        ord('z'):         ( 0,  0,  0,  0, -1,  0,  0),  # disengage the controller
        ord('m'):         ( 0,  0,  0,  0,  0,  0,  1),  # engage the attacker 
        ord('n'):         ( 0,  0,  0,  0,  0,  0, -1)   # disengage the attacker
    }

    def run(self):
        rate = rospy.Rate(self._hz)
        self._running = True
        while self._running:
            while True:
                keycode = self._interface.read_key()
                if keycode is None:
                    break
                self._key_pressed(keycode)
            self._set_velocity()
            self._publish()
            rate.sleep()

    def _get_velcmd(self, linear, angular):
        velcmd = Twist()
        velcmd.linear.x = linear[0]
        velcmd.linear.y = linear[1]
        velcmd.linear.z = linear[2]
        velcmd.angular.x = angular[0]
        velcmd.angular.y = angular[1]
        velcmd.angular.z = angular[2]
        return velcmd

    def _set_velocity(self):
        now = rospy.get_time()
        keys = []
        for a in self._last_pressed:
            if now - self._last_pressed[a] < 0.4:
                keys.append(a)
        
        vx, vy, vz, wx, wy, wz = (0, 0, 0, 0, 0, 0)
        attack_on_off = 0

        for k in keys:
            vx, vy, vz, wx, wy, wz, attack_on_off = self.movement_bindings[k]

        if vx > 0:
            vx = vx * self._forward_rate
        else:
            vx = vx * self._backward_rate

        vy = vy * self._side_rate
        vz = vz * self._elevation_rate
        wz = wz * self._rotation_rate * 5

        self._linear = (vx, vy, vz)
        self._angular = (wx, wy, wz)
        self._attack_on_off = attack_on_off
        

    def _key_pressed(self, keycode):
        print('preseed key:', keycode)
        if keycode == ord('o'):
            self._running = False
            rospy.signal_shutdown('Bye')
        elif keycode in self.movement_bindings:
            self._last_pressed[keycode] = rospy.get_time()

    def _publish(self):
        self._interface.clear()
        self._interface.write_line(1, 'Foward/Backward: %f' % (self._linear[0]))
        self._interface.write_line(2, 'Left/Right     : %f' % (self._linear[1]))
        self._interface.write_line(3, 'Up/Down        : %f' % (self._linear[2]))
        self._interface.write_line(4, 'Angular        : %f' % (self._angular[2]))
        self._interface.write_line(6, 'Use arrow keys to move, e to takeoff, and q to land.')
        self._interface.write_line(7, 'To use the controller, c to engage, and z to disengage.')
        self._interface.write_line(8, 'To add the img attack, m to engage, and n to disengage.')
        self._interface.write_line(9, 'Use o to close the window.')
        self._interface.refresh()

        velcmd = self._get_velcmd(self._linear, self._angular)
        self._pub_cmd.publish(velcmd)

        
        if self._attack_on_off == 1:
            print('attack on!')
            self._image_attack_bool.data = True
        elif self._attack_on_off == -1:
            print('attack off!')
            self._image_attack_bool.data = False

        self._pub_cmd_attack.publish(self._image_attack_bool)


def main(stdscr):
    rospy.init_node('key_teleop')
    app = SimpleKeyTeleop(TextWindow(stdscr))
    app.run()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass