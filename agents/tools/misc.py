#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math

import numpy as np

import carla
from config import cfg


def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    # return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)        # 3.6 * meter per seconds = kmh
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)  # meter per seconds





def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return d_angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2
    location_1, location_2:   carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]

def key_actor(actors_batch, fpath, ego, ego_s):
    Tc = 1.0
    safe_s_ahead = get_speed(ego) * 3.6
    car_length = float(cfg.CARLA.CAR_LENGTH)
    car_width = float(cfg.CARLA.CAR_WIDTH)
    actor_ahead = []
    actor_follow = []
    vehicle_ahead = []
    vehicle_follow = []

    for i, actor in enumerate(actors_batch):
        actor['delta_s'] = actor['Obj_Frenet_state'][0] - ego_s

    for i, actor in enumerate(actors_batch):
        actor_ahead_a = False
        for j in range(len(fpath.s)):
            if abs(actor['Obj_Frenet_state'][0] - fpath.s[j]) <= car_length \
                    and abs(actor['Obj_Frenet_state'][1] - fpath.d[j]) <= car_width \
                    and actor['Obj_Frenet_state'][0] > ego_s:
                actor['safe_acc'] = 2 * (
                        actor['delta_s'] - safe_s_ahead + Tc * (get_speed(actor['Actor']) - get_speed(ego))) / (
                                            Tc ** 2)
                actor_ahead.append(actor)
                actor_ahead_a = True
                break
        if not actor_ahead_a:
            for j in range(len(fpath.s)):
                if fpath.s[j] - actor['Obj_Frenet_state'][0] > 0 and abs(
                        actor['Obj_Frenet_state'][1] - fpath.d[j]) <= car_width:
                    safe_s_follow = get_speed(actor['Actor']) * 3.6
                    # actor['safe_acc'] = 2 * (actor['delta_s'] + safe_s_follow + Tc * (
                    #         get_speed(actor['Actor']) - get_speed(ego))) / (Tc ** 2)
                    actor['safe_acc'] = 2 * (actor['delta_s'] + safe_s_follow - Tc * abs(
                            get_speed(actor['Actor']) - get_speed(ego))) / (Tc ** 2)
                    actor['s_path_obj'] = fpath.s[j] - actor['Obj_Frenet_state'][0]
                    actor_follow.append(actor)
                    break

    # vehicle_ahead 关键前车
    if actor_ahead:
        min_delta_s = min(actor_ahead[i]['delta_s'] for i in range(len(actor_ahead)))
        for i, actor in enumerate(actor_ahead):
            if actor['delta_s'] == min_delta_s:
                vehicle_ahead = actor
                if vehicle_ahead['delta_s'] > safe_s_ahead:
                    vehicle_ahead = None
                break
    # vehicle_follow 关键后车
    if actor_follow:
        min_s_path_obj = min(actor_follow[i]['s_path_obj'] for i in range(len(actor_follow)))
        for i, actor in enumerate(actor_follow):
            if actor['s_path_obj'] == min_s_path_obj:
                safe_s_follow = get_speed(actor['Actor']) * 3.6
                vehicle_follow = actor
                if vehicle_follow['delta_s'] < -safe_s_follow:
                    vehicle_follow = None
                break

    return vehicle_ahead, vehicle_follow