import carla
import lxml.etree as ET
import json

client = carla.Client('localhost', 2000)

client.set_timeout(20.0)

client.load_world("Town04")

world = client.get_world()

world.get_spectator().set_location(carla.Location(x=-25.763199, y=282.871613, z=10))
map = world.get_map()
waypoints = map.generate_waypoints(5)
intersections = []
non_intersections = []
counter = 0
for waypoint in waypoints:
    if waypoint.get_right_lane() and waypoint.get_right_lane().lane_type == carla.LaneType.Driving:
        if waypoint.is_intersection:
            world.debug.draw_string(waypoint.transform.location, text="Intersection", color=carla.Color(255, 0, 0),
                                    life_time=100)
            intersections.append(waypoint)
        else:
            #world.debug.draw_string(waypoint.transform.location, text="Not", color=carla.Color(0, 255, 0),
            #                        life_time=100)
            non_intersections.append(waypoint)

boundary = []

for intersection in intersections:
    next_waypoint = intersection.next(5)[0]
    if not next_waypoint.is_intersection:
        #world.debug.draw_string(intersection.transform.location, text="Boundary", color=carla.Color(0, 0, 255),
        #                        life_time=100)
        boundary.append(next_waypoint)

def get_waypoint_in_distance(waypoint, distance):

    traveled_distance = 0
    while not waypoint.is_intersection and traveled_distance < distance:
        waypoint_new = waypoint.next(1.0)[-1]
        traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
        waypoint = waypoint_new

    return waypoint, traveled_distance

def in_epsilon_distance(value1, value2, epsilon=2):
    return abs(value1 - value2) < epsilon

first_vehicle_location = 25
second_vehicle_location = 25 + 41

route_scenario = []

for boundary_point in boundary:
    route_start = boundary_point.previous(10)[0]
    world.debug.draw_string(boundary_point.transform.location, text="Scenario_Start", color=carla.Color(0, 0, 255),
                            life_time=100)
    world.debug.draw_string(route_start.transform.location, text="Route_Start", color=carla.Color(0, 0, 255),
                           life_time=100)
    first_actor_waypoint,dis1 = get_waypoint_in_distance(boundary_point, first_vehicle_location)

    second_actor_waypoint,dis2 = get_waypoint_in_distance(boundary_point, second_vehicle_location)
    if not in_epsilon_distance(first_vehicle_location, dis1) or not in_epsilon_distance(second_vehicle_location, dis2):
        world.debug.draw_string(first_actor_waypoint.transform.location, text="First Actor",
                                color=carla.Color(255, 255, 0),
                                life_time=100)
        world.debug.draw_string(second_actor_waypoint.transform.location, text="Second Actor",
                                color=carla.Color(255, 255, 0),
                                life_time=100)
        continue
    world.debug.draw_string(first_actor_waypoint.transform.location, text="First Actor", color=carla.Color(0, 255, 0),
                            life_time=100)
    world.debug.draw_string(second_actor_waypoint.transform.location, text="Second Actor", color=carla.Color(0, 255, 0),
                            life_time=100)
    route_end = second_actor_waypoint.next(100)[0]
    world.debug.draw_string(route_end.transform.location, text="Route_End", color=carla.Color(0, 0, 255),
                            life_time=100)
    route_scenario.append((route_start, boundary_point, route_end))

workspace = "/home/simon/Documents/Studium/8.Semester/Bachelorarbeit_Autonomes_Fahren/transfuser_github/transfuser/leaderboard/data/training"
output_scenario = workspace + "/scenarios/Scenario2/Town04_Scenario2.json"
output_route = workspace + "/routes/Scenario2/Town04_Scenario2.xml"

town = "Town04"

route_id = 0

root = ET.Element('routes')

dict = {
        "available_scenarios": [
            {
                town: [
                    {
                        "available_event_configurations": [

                        ],
                        "scenario_type": "Scenario2"
                    }
                ]
            }
        ]
    }

for route_start, boundary_point, route_end in route_scenario:
    route = ET.SubElement(root, 'route', id='%d' % route_id, town=town)
    ET.SubElement(route, 'waypoint',
                  x=str(route_start.transform.location.x),
                  y=str(route_start.transform.location.y), z='0.0',
                  pitch='0.0', roll='0.0',
                  yaw=str(route_start.transform.rotation.yaw))
    ET.SubElement(route, 'waypoint',
                  x=str(route_end.transform.location.x),
                  y=str(route_end.transform.location.y), z='0.0',
                  pitch='0.0', roll='0.0',
                  yaw=str(route_end.transform.rotation.yaw))
    route_id += 1

    dict["available_scenarios"][0]["Town04"][0]["available_event_configurations"].append({
        "transform": {
            "x": boundary_point.transform.location.x,
            "y": boundary_point.transform.location.y,
            "z": boundary_point.transform.location.z,
            "yaw": boundary_point.transform.rotation.yaw,
            "pitch": boundary_point.transform.rotation.pitch
        }
    })

with open(output_scenario, 'w') as f:
    json.dump(dict, f, indent=2, sort_keys=True)

tree = ET.ElementTree(root)

tree.write(output_route, xml_declaration=True, encoding='utf-8', pretty_print=True)

