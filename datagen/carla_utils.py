import carla

class CarlaHandler:
    def __init__(self, server_ip, server_port, world_name):
        """
        Initializes the CarlaHandler class by connecting to the CARLA server and loading the specified world.

        Args:
        - server_ip (str): IP address of the CARLA server.
        - server_port (int): Port on which the CARLA server is running.
        - world_name (str): Name of the desired world to be loaded in CARLA.
        """
        # Create a client to connect to the CARLA server
        self.client = carla.Client(server_ip, server_port)

        # Set a timeout for the client-server connection (in seconds)
        self.client.set_timeout(20.0)

        # Get the current world (map) from the CARLA server
        self.world = self.client.get_world()

        # If the current world is not the desired world, load the correct world
        if world_name != self.world.get_map().name:
            self.client.load_world(world_name)
            # Update the world object with the newly loaded world
            self.world = self.client.get_world()

    def draw_waypoints(self, waypoints, color, debug=False):
        """
        Draws waypoints on the CARLA map using the debug drawing system.

        Args:
        - waypoints (list): A list of waypoints to be drawn.
        - color (carla.Color): Color in which the waypoints will be drawn.
        - debug (bool): If True, additional information (such as whether the waypoint is an intersection) will be displayed.
        """
        # Initialize index to keep track of waypoint number
        index = 0

        color = carla.Color(color[0], color[1], color[2])

        # Iterate through the list of waypoints
        for waypoint in waypoints:
            if debug:
                # If debug is True, display additional information
                if waypoint.is_intersection:
                    # Mark intersection waypoints in red and display index
                    self.world.debug.draw_string(waypoint.transform.location, text=f"Intersection - {index}",
                                                 color=carla.Color(255, 0, 0, 255), life_time=100)
                else:
                    # Mark non-intersection waypoints in green and display index
                    self.world.debug.draw_string(waypoint.transform.location, text=f"Not - {index}",
                                                 color=carla.Color(0, 255, 0, 255), life_time=100)
            else:
                # If debug is False, just display the waypoints with a simple label "W"
                self.world.debug.draw_string(waypoint.transform.location, text="W", color=color, life_time=100)

            # Increment the waypoint index
            index += 1

    def get_waypoint(self, location):
        return self.world.get_map().get_waypoint(carla.Location(location[0], location[1], location[2]))


if __name__ == '__main__':
    # Example usage: create a CarlaHandler instance and connect to the CARLA server at 'localhost' on port 2000.
    # It will load the map 'Town04'.
    handler = CarlaHandler('localhost', 2000, "Town04")