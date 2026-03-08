#SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
# Imports
import math
import numpy as np
from controllers.acc import AccControl
from controllers.lateral_control import LateralControl

param map = localPath('../CARLA/Town06.xodr')
param carla_map = 'Town06'
param time_step = 1.0/10
model scenic.simulators.metadrive.model
param verifaiSamplerType = 'halton'

#CONSTANTS
TERMINATE_TIME = 20 / globalParameters.time_step
POLITENESS_FACTOR = 0.1  # Value between 0 (selfish) and 1 (considerate)
SAFE_BRAKING = -1.0      # Max deceleration for follower in m/s^2
ACCEL_THRESHOLD = 0.1    # Min acceleration gain needed to make a change

def not_zero(x: float, eps: float = 1e-2) -> float:
	if abs(x) > eps:
		return x
	elif x > 0:
		return eps
	else:
		return -eps

def get_vehicle_ahead(id, vehicle, lane, thresholdDistance=25.0):
	""" Returns the closest object in front of the vehicle that is:
	(1) visible,
	(2) on the same lane (or intersection),
	(3) within the thresholdDistance.
	Returns the object if found, or None otherwise. """
	closest = None
	minDistance = float('inf')
	objects = simulation().objects
	for obj in objects:
		if not (vehicle can see obj):
			continue
		d = (distance from vehicle.position to obj.position)
		if d < 0.1:
			continue
		if lane != obj._lane:
			continue
		if d < minDistance and d < thresholdDistance:
			minDistance = d
			closest = obj
	return closest

def get_vehicle_behind(id, vehicle, lane, thresholdDistance=25.0):
	""" Returns the closest object behind the vehicle that is:
	(1) visible,
	(2) on the same lane (or intersection),
	(3) within the thresholdDistance.
	Returns the object if found, or None otherwise. """
	closest = None
	minDistance = float('inf')
	objects = simulation().objects
	for obj in objects:
		if not (obj can see vehicle):
			continue
		d = abs(obj.position.x - vehicle.position.x)
		if vehicle == obj or d < 0.1:
			continue
		if lane != obj._lane:
			continue
		if d < minDistance and d < thresholdDistance:
			minDistance = d
			closest = obj
	return closest

def get_adjacent_lane(id, vehicle, direction):
	"""Get the adjacent lane in the specified direction ('left' or 'right') from the current lane."""
	lane_section = network.laneSectionAt(vehicle.position)
	if lane_section is None:
		return None

	if direction == "left" and lane_section._laneToLeft:
		return lane_section.laneToLeft.lane
	if direction == "right" and lane_section._laneToRight:
		return lane_section.laneToRight.lane

	return None

def check_safety_criterion(current_car, new_follower, long_control):
	if new_follower is None:
		return True
	vehicles_for_control = [current_car, new_follower]
	_, _, new_follower_a = long_control.compute_control(vehicles_for_control, mobil=True)
	return new_follower_a >= SAFE_BRAKING

def check_incentive_criterion(current_car, new_follower, new_leader, long_control):
	# Current car's current acceleration
	accel_self_before = current_car.metaDriveActor.throttle_brake

	# New follower's current acceleration
	accel_new_follower_before = 0
	if new_follower:
		accel_new_follower_before = new_follower.metaDriveActor.throttle_brake

	accel_self_after = long_control.calculate_hypothetical_acceleration(current_car, new_leader)

	# new follower's acceleration after lane change
	accel_new_follower_after = 0
	if new_follower:
		accel_new_follower_after = long_control.calculate_hypothetical_acceleration(new_follower, current_car)

	my_gain = accel_self_after - accel_self_before
	their_loss = accel_new_follower_before - accel_new_follower_after

	if my_gain > (POLITENESS_FACTOR * their_loss) + ACCEL_THRESHOLD:
		return True
	else:
		return False

# Victim behavior
behavior ACC_MOBIL(id, dt, ego_speed, lane):
	thresholdDistance = 25.0
	intervehicle_distance = 7
	long_control = AccControl(id, dt, ego_speed, False, intervehicle_distance)
	long_control_mobil = AccControl(id, dt, ego_speed, False, intervehicle_distance)
	lat_control  = LateralControl(dt)
	was_changing_lanes = False
	while True:
		vehicle_front = None
		current_lane = network.laneAt(self)
		# find vehicle in front
		vehicle_front = get_vehicle_ahead(id, self, current_lane)

		vehicles_for_control = []
		if vehicle_front:
			vehicles_for_control.append(vehicle_front)
		vehicles_for_control.append(self)

		# Lateral: MOBIL
		best_change_advantage = -float('inf')
		target_lane_for_change = None
		lat_control.set_target(current_lane, 'keeping') # default to keeping
		if vehicle_front:
			for direction in ["left", "right"]:
				adjacent_lane = get_adjacent_lane(id, self, direction)
				if adjacent_lane is None or adjacent_lane == current_lane:
					continue

				# in current lane
				old_leader = get_vehicle_ahead(id, self, current_lane)
				old_follower = get_vehicle_behind(id, self, current_lane)

				# in adjacent lane
				adjacent_leader = get_vehicle_ahead(id, self, adjacent_lane)
				adjacent_follower = get_vehicle_behind(id, self, adjacent_lane)

				is_safe = check_safety_criterion(self, adjacent_follower, long_control_mobil)
				is_worth_it = check_incentive_criterion(self, adjacent_follower, adjacent_leader, long_control_mobil)

				if is_safe and is_worth_it:
					target_lane_for_change = adjacent_lane
					lat_control.set_target(target_lane_for_change, 'changing')
					was_changing_lanes = True
				else:
					lat_control.set_target(current_lane, 'keeping')
		### end lane changing logic

		s = lat_control.compute_control(self)
		if was_changing_lanes and lat_control.mode == 'keeping':
			# If so, reset the longitudinal controller to erase its stale state.
			long_control.reset()
			lat_control.reset()
			was_changing_lanes = False

		if lat_control.mode == 'changing':
			b, t = 0.5, 1
		else:
			vehicles_for_control = []
			if vehicle_front:
				vehicles_for_control.append(vehicle_front)
			vehicles_for_control.append(self)
			b, t, _ = long_control.compute_control(vehicles_for_control)
		take SetThrottleAction(t), SetBrakeAction(b), SetSteerAction(s)

# Attack params
amplitude_brake = VerifaiRange(0, 1)
amplitude_acc   = VerifaiRange(0, 1)
amplitude_steer = VerifaiRange(0, 1)
frequency 		= VerifaiRange(0, 10)
attack_time 	= VerifaiRange(0, 10)
duty_cycle      = VerifaiRange(0, 1)

# Attack Behavior
behavior Attacker(id, dt, ego_speed, lane):
	attack_params = {
		'amplitude_brake': amplitude_brake,
		'amplitude_acc': amplitude_acc,
		'amplitude_steer': amplitude_steer,
		'frequency': frequency,
		'attack_time': attack_time,
		'duty_cycle': duty_cycle,
	}
	intervehicle_distance = 7
	long_control = AccControl(id, dt, ego_speed, True, intervehicle_distance, attack_params)
	lat_control  = LateralControl(globalParameters.time_step)
	time_elapsed = 0
	while True:
		vehicles_for_control = [self] + victim_vehicles
		b, t, _ = long_control.compute_control(vehicles_for_control)
		s = math.sin(time_elapsed * frequency) * amplitude_steer # sine wave steering with amplitude 0.3
		time_elapsed += dt
		take SetThrottleAction(t), SetBrakeAction(b), SetSteerAction(s)

#MONITORS
monitor Rewarder(ego_car, victims, goal_region):
	"""Calculates and assigns the reward at each time step."""
	# --- 1. DEFINE REWARD/PENALTY VALUES ---
	# These are the big rewards given once at the end of an episode.
	GOAL_REWARD = 10.0                 # For achieving the specific crash
	EGO_CRASH_PENALTY = -5.0            # For the ego agent crashing (worst outcome)

	# --- 2. DEFINE SHAPING WEIGHTS ---
	forward_weight = 1.0
	speed_weight = 1.0

	# Initialize last_position so first step has progress 0
	if not hasattr(ego_car, 'last_position'):
		ego_car.last_position = ego_car.position
	while True:
		# Check for termination conditions first
		victim_crash = any(v.metaDriveActor.crash_vehicle or v.metaDriveActor.crash_object for v in victims)
		victim_offroad = any(not v._lane for v in victims)
		ego_crashed = ego_car.metaDriveActor.crash_vehicle or ego_car.metaDriveActor.crash_object
		ego_off_road = not ego_car._lane
		all_passed = all(v.position.x > ego_car.position.x for v in victims)
		long_now = ego_car.position[0]
		long_last = ego_car.last_position[0]
		progress = long_now - long_last

		# Set termination reward only once using the object's attribute
		if (victim_crash or victim_offroad) and (not ego_crashed and not ego_off_road):
			ego_car.reward = GOAL_REWARD
			terminate
		elif ego_off_road or ego_crashed or all_passed:
			ego_car.reward = EGO_CRASH_PENALTY
			terminate
		else:
			# dense rewards (only if no termination condition)
			reward = 0.0

			reward += forward_weight * progress
			reward += speed_weight * (ego_car.metaDriveActor.speed_km_h / ego_car.metaDriveActor.max_speed_km_h) * np.sign(progress)

			ego_car.reward = reward
			ego_car.last_position = ego_car.position
			wait

behavior dummy_behavior():
	while True:
		take SetThrottleAction(0.0), SetBrakeAction(0.0), SetSteerAction(0.0)

#PLACEMENT
ego_spawn_pt  = (200 @ -244.5)
victim_spawn_pt = (120 @ -244.5)
num_vehicles_to_place = 5
lane_width = 3.5

id = 0
ego = new Car on ego_spawn_pt #, with behavior Attacker(id, globalParameters.time_step, 10, ego_spawn_pt)

lane_group = network.laneGroupAt(victim_spawn_pt)
victim_vehicles = []
for i in range(num_vehicles_to_place):
	follower_id = i + 1
	lane_i = Uniform(*lane_group.lanes)
	c_i = new Car on lane_i, with behavior ACC_MOBIL(follower_id, globalParameters.time_step, 15, lane_i)
	victim_vehicles.append(c_i)

def true_dist(car1, car2):
	bp1 = car1._boundingPolygon
	bp2 = car2._boundingPolygon
	return bp1.distance(bp2)

MIN_SPACING = 2.0
for i in range(num_vehicles_to_place):
	for j in range(i + 1, num_vehicles_to_place):
		v1 = victim_vehicles[i]
		v2 = victim_vehicles[j]
		require true_dist(v1, v2) > MIN_SPACING

goal_region = RectangularRegion((360.0,-244.5,0), 0, 60, 60)

def inside_lane(car):
	if car._lane is None:
		return 0
	left_edge = car._lane.leftEdge
	right_edge = car._lane.rightEdge
	dist_to_left, dist_to_right = left_edge.signedDistanceTo(car.position), right_edge.signedDistanceTo(car.position)
	half_width = car.width / 2.0
	return int(abs(dist_to_left) < 2 and abs(dist_to_right) < 2)		

# require monitor EndSimulation(ego, victim_vehicles)
record ego.speed as v0_speed
record victim_vehicles[0].speed as v1_speed
record victim_vehicles[1].speed as v2_speed
record victim_vehicles[2].speed as v3_speed
record victim_vehicles[3].speed as v4_speed
record victim_vehicles[4].speed as v5_speed

record inside_lane(ego) as v0_inlane
record inside_lane(victim_vehicles[0]) as v1_inlane
record inside_lane(victim_vehicles[1]) as v2_inlane
record inside_lane(victim_vehicles[2]) as v3_inlane
record inside_lane(victim_vehicles[3]) as v4_inlane
record inside_lane(victim_vehicles[4]) as v5_inlane

record true_dist(ego, victim_vehicles[0]) as v0_v1
record true_dist(ego, victim_vehicles[1]) as v0_v2
record true_dist(ego, victim_vehicles[2]) as v0_v3
record true_dist(ego, victim_vehicles[3]) as v0_v4
record true_dist(ego, victim_vehicles[4]) as v0_v5
record true_dist(victim_vehicles[0], victim_vehicles[1]) as v1_v2
record true_dist(victim_vehicles[0], victim_vehicles[2]) as v1_v3
record true_dist(victim_vehicles[0], victim_vehicles[3]) as v1_v4
record true_dist(victim_vehicles[0], victim_vehicles[4]) as v1_v5
record true_dist(victim_vehicles[1], victim_vehicles[2]) as v2_v3
record true_dist(victim_vehicles[1], victim_vehicles[3]) as v2_v4
record true_dist(victim_vehicles[1], victim_vehicles[4]) as v2_v5
record true_dist(victim_vehicles[2], victim_vehicles[3]) as v3_v4
record true_dist(victim_vehicles[2], victim_vehicles[4]) as v3_v5
record true_dist(victim_vehicles[3], victim_vehicles[4]) as v4_v5

record final ego.reward
require monitor Rewarder(ego, victim_vehicles, goal_region)
