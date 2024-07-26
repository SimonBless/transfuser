import carla
import lxml.etree as ET
import json
import random

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

trigger_distance = 25

route_scenario = []

for boundary_point in boundary:
    route_start = boundary_point.previous(10)[0]
    #world.debug.draw_string(boundary_point.transform.location, text="Scenario_Start", color=carla.Color(0, 0, 255),
    #                        life_time=100)
    world.debug.draw_string(route_start.transform.location, text="Route_Start", color=carla.Color(0, 0, 255),
                           life_time=100)
    trigger_waypoint,dis1 = get_waypoint_in_distance(boundary_point, trigger_distance)

    #second_actor_waypoint,dis2 = get_waypoint_in_distance(boundary_point, second_vehicle_location)
    if not in_epsilon_distance(trigger_distance, dis1):
        world.debug.draw_string(trigger_waypoint.transform.location, text="Trigger Point",
                                color=carla.Color(255, 255, 0),
                                life_time=100)
        continue

    world.debug.draw_string(trigger_waypoint.transform.location, text="Trigger Point", color=carla.Color(0, 255, 0),
                            life_time=100)
    #world.debug.draw_string(second_actor_waypoint.transform.location, text="Second Actor", color=carla.Color(0, 255, 0),
    #                        life_time=100)
    route_end = trigger_waypoint.next(200)[0]
    world.debug.draw_string(route_end.transform.location, text="Route_End", color=carla.Color(0, 0, 255),
                            life_time=100)

    if trigger_waypoint.get_left_lane() and trigger_waypoint.get_left_lane().lane_type == carla.LaneType.Driving:
        rand = random.randint(0,1)
        if rand == 0:
            second_actor = (trigger_waypoint.get_right_lane(), "right")
        else:
            second_actor = (trigger_waypoint.get_left_lane(), "left")
    else:
        second_actor = (trigger_waypoint.get_right_lane(), "right")
    world.debug.draw_string(second_actor[0].transform.location, text="Second Actor", color=carla.Color(0, 255, 0),
                            life_time=100)

    route_scenario.append((route_start, trigger_waypoint, second_actor, route_end))


workspace = "/home/simon/Documents/Studium/8.Semester/Bachelorarbeit_Autonomes_Fahren/transfuser_github/transfuser/leaderboard/data/training"
output_scenario = workspace + "/scenarios/Scenario11/Town04_Scenario11.json"
output_route = workspace + "/routes/Scenario11/Town04_Scenario11.xml"

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
                        "scenario_type": "Scenario11"
                    }
                ]
            }
        ]
    }

for route_start, boundary_point, second_actor, route_end in route_scenario:
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

    second_actor_waypoint, direction = second_actor

    dict["available_scenarios"][0]["Town04"][0]["available_event_configurations"].append({
        "transform": {
            "x": boundary_point.transform.location.x,
            "y": boundary_point.transform.location.y,
            "z": boundary_point.transform.location.z,
            "yaw": boundary_point.transform.rotation.yaw,
            "pitch": boundary_point.transform.rotation.pitch
        },
        "other_actors": {
            direction: [
                {
                    "x": second_actor_waypoint.transform.location.x,
                    "y": second_actor_waypoint.transform.location.y,
                    "yaw": second_actor_waypoint.transform.rotation.yaw,
                    "z": -100,
                    "model": "vehicle.tesla.model3"
                }
            ]
        }
    })

with open(output_scenario, 'w') as f:
    json.dump(dict, f, indent=2, sort_keys=True)

tree = ET.ElementTree(root)

tree.write(output_route, xml_declaration=True, encoding='utf-8', pretty_print=True)

