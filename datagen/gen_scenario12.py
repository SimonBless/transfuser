import carla
import lxml.etree as ET
import json
import random


def get_waypoint_in_distance(waypoint, distance):
    """
    Traverse waypoints from the given waypoint until an intersection is reached
    or the specified distance is traveled.

    Args:
        waypoint (carla.Waypoint): The starting waypoint.
        distance (float): The target distance to travel.

    Returns:
        tuple: A tuple containing the final waypoint and the total distance traveled.
    """
    traveled_distance = 0
    while not waypoint.is_intersection and traveled_distance < distance:
        waypoint_new = waypoint.next(1.0)[-1]
        traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
        waypoint = waypoint_new

    return waypoint, traveled_distance


def in_epsilon_distance(value1, value2, epsilon=2):
    """
    Check if two values are within a specified epsilon distance.
    """
    return abs(value1 - value2) < epsilon


def main():
    """
    Main function to generate driving scenarios (Change-Lane) in the CARLA simulator.
    It identifies intersections, sets up trigger points, and creates routes
    for fast and slow vehicles, optionally including lane changes.
    """
    client = carla.Client('localhost', 2000)

    client.set_timeout(20.0)

    client.load_world("Town04")

    world = client.get_world()

    world.get_spectator().set_location(carla.Location(x=369.2245178222656, y=4.388940334320068, z=10))
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
                non_intersections.append(waypoint)

    boundary = []

    for intersection in intersections:
        next_waypoint = intersection.next(2)[0]
        if not next_waypoint.is_intersection:
            boundary.append(next_waypoint)

    trigger_distance = 2
    fast_vehicle_distance = 15
    slow_vehicle_distance = 80

    route_scenario = []

    for boundary_point in boundary:

        route_start = boundary_point.previous(10)[0]

        lane_change = 0.3 < random.random()

        world.debug.draw_string(route_start.transform.location, text="Route_Start", color=carla.Color(0, 0, 255),
                                life_time=100)

        trigger_waypoint, dis1 = get_waypoint_in_distance(boundary_point, trigger_distance)
        if not in_epsilon_distance(trigger_distance, dis1):
            world.debug.draw_string(trigger_waypoint.transform.location, text="Trigger Point",
                                    color=carla.Color(255, 255, 0),
                                    life_time=100)
            continue

        if lane_change:
            trigger_waypoint = trigger_waypoint.get_right_lane()

        world.debug.draw_string(trigger_waypoint.transform.location, text="Trigger Point", color=carla.Color(0, 255, 0),
                                life_time=100)

        fast_vehicle, dis1 = get_waypoint_in_distance(trigger_waypoint, fast_vehicle_distance)
        if not in_epsilon_distance(fast_vehicle_distance, dis1):
            world.debug.draw_string(fast_vehicle.transform.location, text="Fast Vehicle",
                                    color=carla.Color(255, 255, 0),
                                    life_time=100)
            continue

        world.debug.draw_string(fast_vehicle.transform.location, text="Fast Vehicle", color=carla.Color(0, 255, 0),
                                life_time=100)

        slow_vehicle, dis1 = get_waypoint_in_distance(trigger_waypoint, slow_vehicle_distance)
        if not in_epsilon_distance(slow_vehicle_distance, dis1):
            world.debug.draw_string(slow_vehicle.transform.location, text="Slow Vehicle",
                                    color=carla.Color(255, 255, 0),
                                    life_time=100)
            continue

        world.debug.draw_string(slow_vehicle.transform.location, text="Slow Vehicle", color=carla.Color(0, 255, 0),
                                life_time=100)

        change_lane_waypoint = slow_vehicle.previous(15)[0]

        route_end = change_lane_waypoint.next(120)[0]
        if lane_change:
            route_end = change_lane_waypoint.get_left_lane().next(120)[0]
            if not route_end:
                continue

        world.debug.draw_arrow(route_start.transform.location, change_lane_waypoint.transform.location,
                               color=carla.Color(0, 255, 255))
        world.debug.draw_arrow(change_lane_waypoint.transform.location, route_end.transform.location,
                               color=carla.Color(0, 255, 255))
        print(route_start.transform.location.distance(change_lane_waypoint.transform.location))
        print(change_lane_waypoint.transform.location.distance(route_end.transform.location))
        route_scenario.append((route_start, trigger_waypoint, change_lane_waypoint, route_end, lane_change))

    output_scenario = "Town04_Scenario12.json"
    output_route = "Town04_Scenario12.xml"

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
                        "scenario_type": "Scenario12"
                    }
                ]
            }
        ]
    }

    for route_start, trigger_waypoint, change_lane_waypoint, route_end, lane_change in route_scenario:
        route = ET.SubElement(root, 'route', id='%d' % route_id, town=town, lane_change=str(lane_change))
        ET.SubElement(route, 'waypoint',
                      x=str(route_start.transform.location.x),
                      y=str(route_start.transform.location.y), z='0.0',
                      pitch='0.0', roll='0.0',
                      yaw=str(route_start.transform.rotation.yaw))
        ET.SubElement(route, 'waypoint',
                      x=str(change_lane_waypoint.transform.location.x),
                      y=str(change_lane_waypoint.transform.location.y), z='0.0',
                      pitch='0.0', roll='0.0',
                      yaw=str(change_lane_waypoint.transform.rotation.yaw))
        ET.SubElement(route, 'waypoint',
                      x=str(route_end.transform.location.x),
                      y=str(route_end.transform.location.y), z='0.0',
                      pitch='0.0', roll='0.0',
                      yaw=str(route_end.transform.rotation.yaw))
        route_id += 1

        dict["available_scenarios"][0]["Town04"][0]["available_event_configurations"].append({
            "transform": {
                "x": trigger_waypoint.transform.location.x,
                "y": trigger_waypoint.transform.location.y,
                "z": trigger_waypoint.transform.location.z,
                "yaw": trigger_waypoint.transform.rotation.yaw,
                "pitch": trigger_waypoint.transform.rotation.pitch
            }
        })

    with open(output_scenario, 'w') as f:
        json.dump(dict, f, indent=2, sort_keys=True)

    tree = ET.ElementTree(root)

    tree.write(output_route, xml_declaration=True, encoding='utf-8', pretty_print=True)


if __name__ == '__main__':
    main()
