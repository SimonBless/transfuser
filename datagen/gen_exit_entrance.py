import random
from carla_utils import CarlaHandler
import file_utils


def main():
    """
    Generates several routes with multiple highway exits and entrances
    """
    carla_handler = CarlaHandler("localhost", 2000, "Town04")

    starting_points = []
    initial_location = carla_handler.get_waypoint((-22.41, 124.192, 0.033))

    waypoint = initial_location.previous(90)[0]
    starting_points.append(waypoint)

    for i in range(160):
        previous = waypoint.next(10)
        waypoint = previous[0]
        starting_points.append(waypoint)

    # Choose 50 random starting positions and generate routes of random length
    num_routes = 50
    min_length = 5
    random.shuffle(starting_points)
    routes = []
    for i in range(num_routes):
        length = int(min_length + random.randint(0, 15))
        waypoint = starting_points[i]
        route = [waypoint]
        for j in range(length):
            previous = waypoint.next(10)
            waypoint = previous[0]
            route.append(waypoint)
        # print(len(route))
        routes.append(route)
        carla_handler.draw_waypoints(route, color=(255, 0, 0), debug=True)

    number = 8
    start = carla_handler.get_waypoint((401.806640625, 39.407188415527344, 0.0))
    start = start.previous(50)[0]
    min_length = 10
    for i in range(number):
        left = random.random()
        waypoint = start
        if left < 0.5:
            waypoint = waypoint.get_left_lane()
        offset = random.randint(-20, 10)
        if offset < 0:
            waypoint = waypoint.previous(abs(offset))[0]
        else:
            waypoint = waypoint.next(abs(offset))[0]
        length = int(min_length + random.randint(0, 15))
        route = [waypoint]
        for i in range(length):
            previous = waypoint.next(10)
            waypoint = previous[0]
            route.append(waypoint)
        routes.append(route)
        # draw_waypoints(previous, world, color=carla.Color(255, 0, 0), debug=True)
        carla_handler.draw_waypoints(route, color=(255, 0, 0), debug=False)

    file_utils.gen_xml_route_file("entrance_exit.xml", routes)


if __name__ == '__main__':
    main()
