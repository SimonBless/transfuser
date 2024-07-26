import carla

client = carla.Client('localhost', 2000)

client.set_timeout(20.0)

client.load_world("Town04")

world = client.get_world()

world.get_spectator().set_location(carla.Location(x=-25.763199, y=282.871613, z=10))
map = world.get_map()
waypoints = map.generate_waypoints(5)
for waypoint in waypoints:
    #if waypoint.get_right_lane() and waypoint.get_right_lane().lane_type == carla.LaneType.Driving:
    #if waypoint.get_right_lane() and waypoint.get_right_lane().lane_type != carla.LaneType.Sidewalk:
    if waypoint.get_right_lane() and waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder:
        world.debug.draw_string(waypoint.transform.location, text="W", color=carla.Color(255, 0, 0),
                                life_time=100)

dict = {}
for waypoint in waypoints:
    if waypoint.get_right_lane():
        dict[waypoint.get_right_lane().lane_type] = ""
print(dict.keys())
