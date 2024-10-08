import file_utils
import carla


def draw_waypoints(waypoints, world, color, debug=False):
    index = 0
    for waypoint in waypoints:
        if debug:
            if waypoint.is_intersection:
                world.debug.draw_string(waypoint.transform.location, text=f"Intersection - {index}",
                                        color=carla.Color(255, 0, 0, 255),
                                        life_time=100)
            else:
                world.debug.draw_string(waypoint.transform.location, text=f"Not - {index}",
                                        color=carla.Color(0, 255, 0, 255),
                                        life_time=100)
        else:
            world.debug.draw_string(waypoint.transform.location, text="W", color=color,
                                    life_time=100)
        index += 1


def generate_routes(output_file, world):
    # Generate a route with highway exit and entrance
    first_route = []
    waypoint = world.get_map().get_waypoint(carla.Location(x=143.675537109375,
                                                           y=6.641528129577637,
                                                           z=10.673199653625488))

    waypoint = waypoint.next(90)[0]
    first_route.append(waypoint)
    for i in range(90):
        previous = waypoint.next(10)
        waypoint = previous[0]
        first_route.append(waypoint)
    draw_waypoints(first_route, world, color=carla.Color(255, 0, 0), debug=True)

    # Generate a route with lane changes
    second_route = []
    waypoint = world.get_map().get_waypoint(carla.Location(
        x=-57.749114990234375, y=338.7492370605469, z=1.435070276260376))
    lane = 1
    offset = 10
    distance = 60
    dir_left = True
    pos = 0
    for i in range(80):
        second_route.append(waypoint)
        previous = waypoint.next(10)
        waypoint = previous[0]
        pos += 10
        if pos == distance:
            if dir_left:
                if lane == 4:
                    dir_left = False
                    lane -= 1
                    waypoint = waypoint.get_right_lane()
                else:
                    waypoint = waypoint.get_left_lane()
                    lane += 1
            else:
                if lane == 1:
                    dir_left = True
                    lane += 1
                    waypoint = waypoint.get_left_lane()
                else:
                    waypoint = waypoint.get_right_lane()
                    lane -= 1
            pos = 0
            distance += offset
    draw_waypoints(second_route, world, color=carla.Color(255, 0, 0), debug=True)

    # Generate a short route with highway exit and entrance and lane change
    third_route = []
    waypoint = world.get_map().get_waypoint(carla.Location(x=-146.87045288085938,
                                                           y=418.5563659667969,
                                                           z=2.0534846782684326))

    distance = 30
    lane_changes = 3
    offset = 10
    pos = 0
    for i in range(80):
        third_route.append(waypoint)
        previous = waypoint.next(10)
        waypoint = previous[0]
        if len(previous) > 1:
            waypoint = previous[1]

        pos += 10
        if lane_changes > 0 and pos == distance:
            waypoint = waypoint.get_right_lane()
            lane_changes -= 1
            pos = 0
            distance += offset
    draw_waypoints(third_route, world, color=carla.Color(255, 0, 0), debug=True)

    # Generate a long route with highway exit and entrance and lane change in dense traffic
    fourth_route = []
    waypoint = world.get_map().get_waypoint(carla.Location(x=220.90,
                                                           y=-395.659,
                                                           z=0.0))
    waypoint = waypoint.previous(30)[0]
    distance = 40
    lane_changes = 2
    offset = 10
    pos = 0
    for i in range(150):
        fourth_route.append(waypoint)
        previous = waypoint.next(10)
        waypoint = previous[0]

        pos += 10
        if lane_changes > 0 and pos == distance:
            waypoint = waypoint.get_left_lane()
            lane_changes -= 1
            pos = 0
            distance += 50
        elif -1 <= lane_changes <= 0 and pos == distance:
            waypoint = waypoint.get_right_lane()
            lane_changes -= 1
            pos = 0
            distance += 50
    fourth_route.append(waypoint)
    draw_waypoints(fourth_route, world, color=carla.Color(255, 0, 0), debug=True)
    file_utils.gen_xml_route_file(output_file, [first_route, second_route, third_route, fourth_route])
    return first_route, second_route, third_route, fourth_route


def generate_scenarios(output_file, world, routes):
    first_route, second_route, third_route, fourth_route = routes
    scenarios = []

    scenario1 = {
        "scenario_type": "Scenario1",
        "trigger_points": []
    }
    scenario2 = {
        "scenario_type": "Scenario2",
        "trigger_points": []
    }
    scenario3 = {
        "scenario_type": "Scenario3",
        "trigger_points": []
    }
    scenario11 = {
        "scenario_type": "Scenario11",
        "trigger_points": []
    }

    # Create scenario1 trigger points
    scenario1["trigger_points"].append((first_route[2], None))
    scenario1["trigger_points"].append((first_route[20], None))
    scenario1["trigger_points"].append((first_route[60], None))
    scenario1["trigger_points"].append((second_route[1], None))
    scenario1["trigger_points"].append((second_route[21], None))
    scenario1["trigger_points"].append((second_route[51], None))
    scenario1["trigger_points"].append((third_route[7], None))
    scenario1["trigger_points"].append((third_route[21], None))
    scenario1["trigger_points"].append((third_route[74], None))

    # Create scenario2 trigger points
    scenario2["trigger_points"].append((first_route[83], None))
    scenario2["trigger_points"].append((second_route[63], None))

    # Create scenario3 trigger points
    scenario3["trigger_points"].append((third_route[60], None))
    scenario3["trigger_points"].append((second_route[76], None))

    # Create scenario11 trigger points
    trigger_point_scenario11 = second_route[41]
    scenario11_second_actor = trigger_point_scenario11.get_right_lane().previous(2)[0]
    scenario11["trigger_points"].append((trigger_point_scenario11, {
        "right": [
            {
                "model": "vehicle.tesla.model3",
                "x": scenario11_second_actor.transform.location.x,
                "y": scenario11_second_actor.transform.location.y,
                "yaw": scenario11_second_actor.transform.rotation.yaw,
                "z": -100
            }
        ]
    }))
    draw_waypoints([scenario11_second_actor], world, color=carla.Color(255, 0, 0), debug=True)
    trigger_point_scenario11 = third_route[28]
    scenario11_second_actor = trigger_point_scenario11.get_left_lane().previous(2)[0]
    scenario11["trigger_points"].append((trigger_point_scenario11, {
        "left": [
            {
                "model": "vehicle.tesla.model3",
                "x": scenario11_second_actor.transform.location.x,
                "y": scenario11_second_actor.transform.location.y,
                "yaw": scenario11_second_actor.transform.rotation.yaw,
                "z": -100
            }
        ]
    }))
    draw_waypoints([scenario11_second_actor], world, color=carla.Color(255, 0, 0), debug=True)

    scenarios.append(scenario1)
    scenarios.append(scenario2)
    scenarios.append(scenario3)
    scenarios.append(scenario11)
    file_utils.gen_json_scenario_file(output_file, scenarios)


def main():
    client = carla.Client('localhost', 2000)

    client.set_timeout(20.0)

    world = client.get_world()
    spec = world.get_spectator().get_location()
    client.load_world("Town04")
    world = client.get_world()
    world.get_spectator().set_location(spec)

    output_route = "evaluation_route.xml"
    routes = generate_routes(output_route, world)

    output_scenario = "evaluation_scenario.json"
    generate_scenarios(output_scenario, world, routes)


if __name__ == '__main__':
    main()
