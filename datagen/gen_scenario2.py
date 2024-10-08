import lxml.etree as ET  # Library for parsing and creating XML files
import json  # Library for handling JSON data
import carla  # CARLA simulator Python API
import math  # Math library for mathematical operations


def get_distance(point1, point2):
    """
    Calculate the Euclidean distance between two CARLA objects.

    Args:
        point1 (carla.Location | carla.Transform | carla.Waypoint): The first point.
        point2 (carla.Location | carla.Transform | carla.Waypoint): The second point.

    Returns:
        float: The calculated distance.

    Raises:
        Exception: If point1 is not of an expected type.
    """
    if isinstance(point1, carla.Location):
        # Distance between two Location objects
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
    elif isinstance(point1, carla.Transform):
        # Distance between two Transform objects based on their locations
        return math.sqrt((point1.location.x - point2.location.x) ** 2 + (point1.location.y - point2.location.y) ** 2)
    elif isinstance(point1, carla.Waypoint):
        # Distance between two Waypoint objects based on their transforms' locations
        return math.sqrt((point1.transform.location.x - point2.transform.location.x) ** 2 +
                         (point1.transform.location.y - point2.transform.location.y) ** 2)
    else:
        # Raise an exception for unexpected types
        raise Exception(f"Unexpected type {type(point1)}")


def sort_route_file(path, output_path):
    """
    Sort and restructure a route XML file for a specific town and save the new XML.

    Args:
        path (str): Path to the input XML route file.
        output_path (str): Path to save the sorted XML route file.
    """
    town = "Town04"  # Define the town name for the routes
    route_id = 0  # Initialize route identifier

    # Parse the input XML file
    tree = ET.parse(path)
    old_root = tree.getroot()

    # Create a new root element for the sorted routes
    root = ET.Element('routes')

    # Iterate over each route in the original XML
    for child in old_root:
        # Create a new 'route' element with id and town attributes
        route = ET.SubElement(root, 'route', id='%d' % route_id, town=town)

        # Iterate over each waypoint in the route
        for waypoint in child:
            # Add waypoint details to the new route
            ET.SubElement(route, 'waypoint',
                          x=waypoint.attrib['x'],
                          y=waypoint.attrib['y'],
                          z='0.0',  # Set z-coordinate to 0.0
                          pitch='0.0',  # Set pitch to 0.0
                          roll='0.0',  # Set roll to 0.0
                          yaw=waypoint.attrib['yaw'])  # Retain yaw from original

        route_id += 1  # Increment the route identifier

    # Create a new ElementTree with the sorted routes
    tree_new = ET.ElementTree(root)

    # Write the new XML to the specified output path with pretty formatting
    tree_new.write(output_path, xml_declaration=True, encoding='utf-8', pretty_print=True)


def generate_change_lane(output_path):
    """
    Generate a combined route XML file with different lane change scenarios.

    Args:
        output_path (str): Path to save the generated XML route file.
    """

    # TODO
    routes_path = ""

    # Define paths to different lane change XML files
    ll = routes_path + "/ll/Town04_ll.xml"  # Left to Left lane change
    lr = routes_path + "/lr/Town04_lr.xml"  # Left to Right lane change
    rl = routes_path + "/rl/Town04_rl.xml"  # Right to Left lane change
    rr = routes_path + "/rr/Town04_rr.xml"  # Right to Right lane change

    town = "Town04"  # Define the town name
    route_id = 0  # Initialize route identifier

    # Create a new root element for the combined routes
    root = ET.Element('routes')

    # List of tuples containing route types and their corresponding file paths
    route_types_files = [
        ("ll", ll),
        ("lr", lr),
        ("rl", rl),
        ("rr", rr)
    ]

    # Iterate over each route type and its file
    for current_route_type, file_path in route_types_files:
        # Parse the current route type XML file
        tree = ET.parse(file_path)
        old_root = tree.getroot()

        # Iterate over each route in the parsed XML
        for child in old_root:
            # Create a new 'route' element with id, town, and type attributes
            route = ET.SubElement(root, 'route',
                                  id='%d' % route_id,
                                  town=town,
                                  type=current_route_type)

            # Iterate over each waypoint in the route
            for waypoint in child:
                # Add waypoint details to the new route
                ET.SubElement(route, 'waypoint',
                              x=waypoint.attrib['x'],
                              y=waypoint.attrib['y'],
                              z='0.0',  # Set z-coordinate to 0.0
                              pitch='0.0',  # Set pitch to 0.0
                              roll='0.0',  # Set roll to 0.0
                              yaw=waypoint.attrib['yaw'])  # Retain yaw from original

            route_id += 1  # Increment the route identifier

    # Create a new ElementTree with the combined routes
    tree_new = ET.ElementTree(root)

    # Write the combined routes XML to the specified output path with pretty formatting
    tree_new.write(output_path, xml_declaration=True, encoding='utf-8', pretty_print=True)


def visualize_waypoint(waypoint, world, boundary_point=False, life_time=100):
    """
    Visualize a waypoint in the CARLA simulation world.

    Args:
        waypoint (carla.Waypoint): The waypoint to visualize.
        world (carla.World): The CARLA simulation world.
        boundary_point (bool, optional): Flag to indicate if the waypoint is a boundary point. Defaults to False.
        life_time (float, optional): Duration in seconds for which the visualization remains. Defaults to 100.
    """
    if boundary_point:
        # Draw a blue string labeled "Boundary" at the waypoint location
        world.debug.draw_string(
            waypoint.transform.location,
            text="Boundary",
            color=carla.Color(0, 0, 255),
            life_time=life_time
        )
    else:
        if waypoint.is_intersection:
            # Draw a red string labeled "Intersection" if the waypoint is an intersection
            world.debug.draw_string(
                waypoint.transform.location,
                text="Intersection",
                color=carla.Color(255, 0, 0),
                life_time=life_time
            )
        else:
            # Draw a green string labeled "Not" if the waypoint is not an intersection
            world.debug.draw_string(
                waypoint.transform.location,
                text="Not",
                color=carla.Color(0, 255, 0),
                life_time=life_time
            )


#path = "/home/simon/Documents/Studium/8.Semester/Bachelorarbeit_Autonomes_Fahren/transfuser_github/transfuser/leaderboard/data/training/routes/highway/highway.xml"
#output_path = "/home/simon/Documents/Studium/8.Semester/Bachelorarbeit_Autonomes_Fahren/transfuser_github/transfuser/leaderboard/data/training/scenarios/Scenario2/Town04_Scenario2_testing.json"
#routes_path = "/home/simon/Documents/Studium/8.Semester/Bachelorarbeit_Autonomes_Fahren/transfuser_github/transfuser/leaderboard/data/training/routes"

#workspace = "/home/simon/Documents/Studium/8.Semester/Bachelorarbeit_Autonomes_Fahren/transfuser_github/transfuser/leaderboard/data/training"
#highway_route = workspace + "/routes/highway/highway.xml"
#output_scenario = workspace + "/scenarios/Scenario2/Town04_Scenario2.json"
#output_route = workspace + "/routes/Scenario2/Town04_Scenario2.xml"


def main():
    """
    Generate Scenario2 configurations based on highway routes and save them as a JSON file.
    """

    highway_route_file = "highway.xml"
    town = "Town04"

    # Print the input file paths for debugging purposes
    print(f"Highway_route: {highway_route_file}")

    # Initialize the dictionary structure for the JSON scenario
    scenario_dict = {
        "available_scenarios": [
            {
                town: [
                    {
                        "available_event_configurations": [
                            # Event configurations will be appended here
                        ],
                        "scenario_type": "Scenario2"  # Define the scenario type
                    }
                ]
            }
        ]
    }

    max_steps = 50  # Maximum number of steps to iterate through waypoints
    epsilon_distance = 10  # Distance threshold to consider reaching the end
    amount_of_way_points = 20  # Number of waypoints to divide the distance

    # Initialize CARLA client and get the world and map
    client = carla.Client('localhost', 2000)
    # client.load_world(town)        # Uncomment if you need to load a specific world
    # time.sleep(10)                 # Wait for the world to load (uncomment if necessary)
    world = client.get_world()
    map = world.get_map()

    # Parse the highway route XML file
    tree = ET.parse(highway_route_file)
    root = tree.getroot()

    # Counters for intersections
    intersection_count = 0
    not_intersection_count = 0

    # Iterate over each route in the highway routes XML
    for route in root:
        waypoints = route.findall("waypoint")  # Find all waypoints in the route
        waypoint_start = waypoints[0]  # Starting waypoint
        waypoint_end = waypoints[1]  # Ending waypoint (assumes at least two waypoints)

        # Convert waypoint attributes to CARLA Location objects
        location_start = carla.Location(
            float(waypoint_start.attrib['x']),
            float(waypoint_start.attrib['y']),
            float(waypoint_start.attrib['z'])
        )

        location_end = carla.Location(
            float(waypoint_end.attrib['x']),
            float(waypoint_end.attrib['y']),
            float(waypoint_end.attrib['z'])
        )

        trigger_points = []  # List to store trigger points along the route

        # Get CARLA Waypoint objects from locations, projecting to road
        waypoint_start = map.get_waypoint(location_start, project_to_road=True)
        visualize_waypoint(waypoint_start, world, boundary_point=True)  # Visualize start waypoint

        waypoint_end = map.get_waypoint(location_end, project_to_road=True)
        visualize_waypoint(waypoint_end, world, boundary_point=True)  # Visualize end waypoint

        # Calculate the distance to move in each step
        distance = get_distance(waypoint_start, waypoint_end) / amount_of_way_points

        waypoint_iterator = waypoint_start  # Initialize the waypoint iterator

        # Iterate through waypoints up to max_steps
        for x in range(0, max_steps):
            waypoints = waypoint_iterator.next(distance)  # Get next waypoint(s) at the specified distance
            # print(f"Length of waypoint array: {len(waypoints)}")  # Debug: print number of waypoints retrieved

            for waypoint_tmp in waypoints:
                visualize_waypoint(waypoint_tmp, world)  # Visualize each waypoint

            if not waypoints:
                break  # Exit if no waypoints are available

            waypoint_iterator = waypoints[0]  # Move to the next waypoint
            trigger_points.append(waypoints[0])  # Add to trigger points

            # Check if the current waypoint is within epsilon_distance of the end location
            if get_distance(location_end, waypoints[0].transform.location) < epsilon_distance:
                break

        if trigger_points:
            pos_fifth = int(len(trigger_points) / 5)  # Calculate one-fifth of the trigger points
            position = pos_fifth

            # Find a waypoint in the middle segment that is not an intersection
            for i in range(pos_fifth, int(len(trigger_points) / 2)):
                if not trigger_points[i].is_intersection:
                    position = i
                    break

            # Update intersection counters based on the selected waypoint
            intersection_count += trigger_points[position].is_intersection
            not_intersection_count += not (trigger_points[position].is_intersection)

            # Get the transform of the selected waypoint
            transform = trigger_points[position].transform

            # Append the transform details to the scenario dictionary
            scenario_dict["available_scenarios"][0]["Town04"][0]["available_event_configurations"].append({
                "transform": {
                    "x": transform.location.x,
                    "y": transform.location.y,
                    "z": transform.location.z,
                    "yaw": transform.rotation.yaw,
                    "pitch": transform.rotation.pitch
                }
            })

    # Save the scenario dictionary to a JSON file with pretty formatting
    with open("Scenario2.json", 'w') as f:
        json.dump(scenario_dict, f, indent=2, sort_keys=True)

    # Print the counts of intersections and non-intersections encountered
    print(f"Intersections: {intersection_count}")
    print(f"Not intersections: {not_intersection_count}")

if __name__ == "__main__":
    main()
