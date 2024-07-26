import carla
import time

client = carla.Client('localhost', 2000)

client.set_timeout(20.0)

client.load_world("Town04")

world = client.get_world()

blueprint_library = world.get_blueprint_library()

vehicle_bp = blueprint_library.find('vehicle.diamondback.century')

world.get_spectator().set_location(carla.Location(x=-25.763199, y=282.871613, z=10))
spawnpoint = carla.Transform(carla.Location(x=-25.763199, y=282.871613, z=0.000000),
                             carla.Rotation(pitch=360.000000, yaw=109.767143, roll=0.000000))

map = world.get_map()
waypoints = map.generate_waypoints(5)
print(carla.Color(255,0,0))
for waypoint in waypoints:
    waypoint.transform.location.z += 5
    if waypoint.is_intersection:
        world.debug.draw_string(waypoint.transform.location, text="Intersection", color=carla.Color(255,0,0,255), life_time=100)
    else:
        world.debug.draw_string(waypoint.transform.location, text="Not",color=carla.Color(0,255,0,255), life_time=100)
