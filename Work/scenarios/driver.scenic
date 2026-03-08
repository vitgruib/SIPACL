param map = localPath('../CARLA/Town05.xodr')
model scenic.simulators.metadrive.model

param time_step = 0.1
param verifaiSamplerType = 'halton'
param render = 1
param use2DMap = True
param realtime = False

param extra_cars = 1
param driving_reward = 1.0
param speed_reward = .1
param debug_print_controls = True
param debug_print_controls_every_steps = 1000
param max_speed_km_h = 80

import numpy as np
TERMINATE_TIME = 40 / globalParameters.time_step

"""
Setting global params for the road, land, starting and stoping
Scene defining variables should be global params in order to allow for
easily mutating later
"""

def get_nearest_centerline(obj):
	min_dist = np.inf
	for lane in network.lanes:
		dist = distance to lane
		if dist < min_dist:
			min_dist = dist
			centerline = lane.centerline
	return centerline

# TODO fix params -- need more variabiltiy and ensure that modified scenes make!


# Random-lane setup:
# start = start point of a random lane
# destination = end point of a random lane
# Hard-coded lane choices for debugging.
start_lane_choice = network.lanes[0]
destination_lane_choice = network.lanes[4]

start_p0 = start_lane_choice.centerline.points[0]
start_p1 = start_lane_choice.centerline.points[-1]
dest_p0 = destination_lane_choice.centerline.points[0]
dest_p1 = destination_lane_choice.centerline.points[-1]

# Pick interior points to reduce instant terminal success at shared lane endpoints.
start_xy = (0.9 * start_p0[0] + 0.1 * start_p1[0], 0.9 * start_p0[1] + 0.1 * start_p1[1], 0)
destination_xy = (0.1 * dest_p0[0] + 0.9 * dest_p1[0], 0.1 * dest_p0[1] + 0.9 * dest_p1[1], 0)

# Export sampled lanes as params after selecting/constraining them.
param start_lane = start_lane_choice
param destination_lane = destination_lane_choice

start = (start_xy[0] @ start_xy[1])
destination_point = (destination_xy[0] @ destination_xy[1])

ego = new Car on start, facing roadDirection, with observation 0, with md_destination_point destination_point
#distractor = new Car on start2, with behavior DriveAvoidingCollisions(target_speed=10, avoidance_threshold=12)

monitor DrivingReward(obj):
	ego.previous_coordinates = obj.position
	step_count = 0
	while True:
		lane = obj._lane

		if lane:
			centerline = lane.centerline	
		else:
			centerline = get_nearest_centerline(obj)

		if obj._lane:
			ego.lane_heading = lane._defaultHeadingAt(ego.position)
			ref_heading = ego.lane_heading
		else:
			ref_heading = ego.heading

		delta = obj.position - ego.previous_coordinates
		driving_progress = delta[0] * np.cos(ref_heading) + delta[1] * np.sin(ref_heading)
		driving_reward = globalParameters.driving_reward * driving_progress

		speed_km_h = ego.speed * 3.6
		speed_reward = globalParameters.speed_reward * (speed_km_h / globalParameters.max_speed_km_h)

		reward = driving_reward + speed_reward

		ego.reward = reward
		ego.reward_speed = speed_reward
		ego.reward_dist = driving_progress

		ego.previous_coordinates = obj.position
		step_count += 1
		wait
	

require monitor DrivingReward(ego)
