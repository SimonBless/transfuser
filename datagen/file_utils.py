import json
import lxml.etree as ET

def gen_json_scenario_file(output_file, scenarios):
    """
    Generate a JSON scenario file.

    Args:
    - output_file (str): output file name
    - scenarios (list): list of scenarios (consisting of dictionaries with scenario_type
                      and list of (trigger_point, list of other actors)
    """

    # Initialize the structure of the output JSON file
    output = {
        "available_scenarios": [
            {
                "Town04": [] # The town in which the scenarios take place
            }
        ]
    }

    for scenario in scenarios:
        scenario_points = {
            "available_event_configurations": [],
            "scenario_type": scenario["scenario_type"]
        }

        for waypoint, other_actors in scenario["trigger_points"]:
            trigger_point = {
                "transform": {
                    "pitch": waypoint.transform.rotation.pitch,
                    "x": waypoint.transform.location.x,
                    "y": waypoint.transform.location.y,
                    "z": waypoint.transform.location.z,
                    "yaw": waypoint.transform.rotation.yaw
                }
            }
            if other_actors:
                trigger_point["other_actors"] = other_actors
            scenario_points["available_event_configurations"].append(trigger_point)
        output["available_scenarios"][0]["Town04"].append(scenario_points)

    # Write the scenarios to the specified output file
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)


def gen_xml_route_file(output_file, routes):
    """
    Generate a XML route file.
    :param output_file: output file name
    :param routes: list of routes (each route consists of a list of waypoints)
    """
    root = ET.Element('routes')
    id = 0
    for route in routes:
        single_route = ET.SubElement(root, 'route', id='%d' % id, town="Town04")
        for waypoint in route:
            ET.SubElement(single_route, 'waypoint',
                      x=str(waypoint.transform.location.x),
                      y=str(waypoint.transform.location.y),
                      z=str(waypoint.transform.location.z),
                      pitch='360.0',
                      roll='0.0',
                      yaw=str(waypoint.transform.rotation.yaw))
        id += 1

    tree = ET.ElementTree(root)

    tree.write(output_file, xml_declaration=True, encoding='utf-8', pretty_print=True)