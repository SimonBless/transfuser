import random
from datetime import datetime
import os

import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import re
import json
import csv
import sys
from os import walk
import seaborn as sns
from PIL import Image

import pygame

# constants
MAP_DIR = "../leaderboard/data/maps"
MAP = "Town04"
PIXELS_PER_METER = 8


def parse_xml(xml_file):
    """
    Parse the XML file containing routes and waypoints.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    routes_data = []
    for route in root.findall('route'):
        waypoints = []
        for waypoint in route.findall('waypoint'):
            x = float(waypoint.get('x'))
            y = float(waypoint.get('y'))
            waypoints.append((x, y))
        routes_data.append(waypoints)

    print("XML data parsed successfully.")
    return routes_data

class CarlaEvaluationProcessor:
    def __init__(self, base_dir_path, output_dir_path):

        self.base_dir = base_dir_path

        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"The directory {self.base_dir} does not exist.")
        if not os.path.isdir(self.base_dir):
            raise NotADirectoryError(f"The path {self.base_dir} is not a directory.")

        self.base_dir = os.path.join(self.base_dir, 'debug')

        #if not os.path.exists(self.debug_dir):
        #    raise FileNotFoundError(f"The debug directory {self.debug_dir} does not exist.")
        #if not os.path.isdir(self.debug_dir):
           # raise NotADirectoryError(f"The path {self.debug_dir} is not a directory.")
        json_name = [f for f in os.listdir(base_dir_path) if f.endswith(".json")][0]
        self.eval_results_file = os.path.join(base_dir_path, json_name)

        if not os.path.isfile(self.eval_results_file):
            raise FileNotFoundError(f"The evaluation result json-file {self.eval_results_file} does not exist.")

        self.num_routes = 4

        #output_path = os.path.join(output_dir_path, 'processed_results')
        self.output_dir = os.path.join(output_dir_path, os.path.basename(base_dir_path))

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=False)
        print(f"Output directory created: {self.output_dir}")

    def create_video(self, route):
        """
        Create a video from the PNG files of a specified route.
        """
        route_dir = os.path.join(self.base_dir, f'route{route}')
        if not os.path.exists(route_dir):
            raise FileNotFoundError(f"The directory {route_dir} does not exist.")
        if not os.path.isdir(route_dir):
            raise NotADirectoryError(f"The path {route_dir} is not a directory.")

        image_pattern = re.compile(r'^(\d+)\.png$')
        #images = [img for img in sorted(os.listdir(route_dir)) if img.endswith(".png") and image_pattern.match(img)]
        sorted_files = sorted(
            (file for file in os.listdir(route_dir) if image_pattern.match(file)),
            key=lambda file: int(image_pattern.match(file).group(1))
        )
        #print(images)
        #print(len(images))

        if not sorted_files:
            print(f"No PNG files found in {route_dir}")
            return

        # Assuming all images have the same dimensions
        frame = cv2.imread(os.path.join(route_dir, sorted_files[0]))
        height, width, layers = frame.shape

        video_path = os.path.join(self.output_dir, f'route{route}_video.mp4')
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        for image in sorted_files:
            video.write(cv2.imread(os.path.join(route_dir, image)))

        video.release()
        print(f"Video saved at {video_path}")

    def load_evaluation_time_data(self):
        """
        Load all CSV speed files and combine them into a single DataFrame.
        """
        all_data = []
        for route_num in range(4):
            route_dir = f"route{route_num}"
            route_path = os.path.join(self.base_dir, route_dir)
            for file in os.listdir(route_path):
                if file.endswith('.csv'):
                    csv_path = os.path.join(route_path, file)
                    names = ['speed','gps','compass','target_x','target_y','position']

                    first_row = pd.read_csv(csv_path, nrows=1).columns.tolist()
                    if any(not str(val).replace('.', '', 1).isdigit() for val in first_row):
                        df = pd.read_csv(csv_path)
                    else:
                        df = pd.read_csv(csv_path, names=names)

                    df['route'] = route_dir  # Add route information to the DataFrame
                    df['gps'] = df['gps'].apply(lambda x: [float(i) for i in x.strip('[]').split()])

                    df[['gps_x', 'gps_y']] = pd.DataFrame(df['gps'].tolist(), index=df.index)

                    # Drop the original 'gps' column
                    df = df.drop(columns=['gps'])

                    df['target_point_x'] = df['target_x'].apply(lambda x: float(x.strip('()')))
                    df['target_point_y'] = df['target_y'].apply(lambda x: float(x.strip('()')))

                    df['speed'] = df['speed'].apply(lambda x: float(x))

                    df['position'] = df['position'].apply(lambda x: [float(i) for i in x.strip('[]').split()])

                    df[['position_x', 'position_y']] = pd.DataFrame(df['position'].tolist(), index=df.index)

                    # Drop the original 'gps' column
                    df = df.drop(columns=['position'])

                    all_data.append(df)

        # Combine all DataFrames into one
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print("Combined CSV DataFrame created.")
            return combined_df, all_data
        else:
            print("No CSV files found.")
            return None



    def plot_map(self, model_data, route_data, output_file_name, model_circle_size=8, route_circle_size=18):
        """
        Plot evaluation data and XML routes together on the respective map.
        """
        pygame.init()
        with open(os.path.join(MAP_DIR, f'{MAP}_details.json'), 'r') as f:
            data = json.load(f)

        world_offset = data['world_offset']
        filename = MAP + "_.tga"
        viz_surface = pygame.image.load(os.path.join(MAP_DIR, filename))

        output_file = os.path.join(self.output_dir, output_file_name)

        for route_name, route_df in model_data.groupby('route'):
            rgb = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            color_pg = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

            for index, row in route_df.iterrows():
                x = PIXELS_PER_METER * (row['world_x'] - world_offset[0])
                y = PIXELS_PER_METER * (row['world_y'] - world_offset[1])
                location = [int(x), int(y)]
                pygame.draw.circle(viz_surface, color_pg, location, model_circle_size)

        rgbs_ = []
        for j, points in enumerate(route_data):
            print(f' Plotted route {j + 1}/{len(route_data)}')
            rgb = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            rgbs_.append(rgb)
            color_pg = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

            # draw points
            for i, point in enumerate(points[:-1]):
                sp_, ep_ = points[i], points[i + 1]

                x = PIXELS_PER_METER * (point[0] - world_offset[0])
                y = PIXELS_PER_METER * (point[1] - world_offset[1])
                location = [int(x), int(y)]
                x = PIXELS_PER_METER * (ep_[0] - world_offset[0])
                y = PIXELS_PER_METER * (ep_[1] - world_offset[1])
                location2 = [int(x), int(y)]

                # draw start point and end point
                pygame.draw.circle(viz_surface, color_pg, location, route_circle_size)

                # draw arrow between sp and ep
                color_white = pygame.Color(255, 255, 255)
                pygame.draw.lines(viz_surface, color_white, False,
                                  [location, location2], 4)

        pygame.image.save_extended(viz_surface, output_file)
        pygame.quit()

    def generate_converted_waypoints(self, evaluation_data, routes_data):
        new_df = evaluation_data.copy()

        new_df['target_point_x'], new_df['target_point_y'] = (
            new_df['target_point_y'], -new_df['target_point_x'])

        new_df['gps_x'], new_df['gps_y'] = (
            new_df['gps_y'], -new_df['gps_x'])

        new_df['position_x'], new_df['position_y'] = (
            new_df['position_y'], -new_df['position_x'])

        routes = new_df.groupby('route').first()
        route_start_locations = []
        model_start_locations = []
        i = 0
        for i in range(4):
            model_start_locations.append((routes['gps_x'][i], routes['gps_y'][i]))
            route_start_locations.append(routes_data[i][0])

        sx1 = (route_start_locations[0][0] - route_start_locations[1][0]) / (
                model_start_locations[0][0] - model_start_locations[1][0])
        sx2 = (route_start_locations[2][0] - route_start_locations[3][0]) / (
                model_start_locations[2][0] - model_start_locations[3][0])
        sy1 = (route_start_locations[0][1] - route_start_locations[1][1]) / (
                model_start_locations[0][1] - model_start_locations[1][1])
        sy2 = (route_start_locations[2][1] - route_start_locations[3][1]) / (
                model_start_locations[2][1] - model_start_locations[3][1])
        sx = (sx1 + sx2) / 2
        sy = (sy1 + sy2) / 2
        bx = 0
        by = 0
        for i in range(4):
            bx += route_start_locations[i][0] - sx * model_start_locations[i][0]
            by += route_start_locations[i][1] - sy * model_start_locations[i][1]
        bx /= 4
        by /= 4

        new_df['world_x'] = sx * new_df['gps_x'] + bx
        new_df['world_y'] = sy * new_df['gps_y'] + by

        def local_to_global(target_point, position, compass):
            """
            Convert a local point to global coordinates.

            target_point: tuple of (x, y) in local coordinates
            position: vehicle's global (x, y) coordinates
            compass: vehicle's heading in radians
            """
            theta = compass + np.pi / 2  # Adjust the compass angle
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            global_point = R.dot(np.array(target_point)) + np.array(position)
            return global_point

        target_points = list(zip(new_df['target_point_x'], new_df['target_point_y']))
        pos = list(zip(new_df['position_x'], new_df['position_y']))
        # Convert all target_points to global coordinates before plotting
        global_target_points = [local_to_global(tp, pos, compass)
                                for tp, pos, compass in zip(target_points, pos, new_df['compass'])]

        # Separate the global points into x and y
        new_df['global_target_x'] = [point[0] for point in global_target_points]
        new_df['global_target_y'] = [point[1] for point in global_target_points]

        return new_df

    def plot_route_specific_index(self, evaluation_data2, y_axis, x_axis, title):
        plt.figure(figsize=(10, 8))
        colors = ['b-', 'g-', 'r-', 'm-']
        i = 1
        for route in evaluation_data2:
            index = range(len(route))
            plt.subplot(2, 2, i)
            plt.plot(index, route, colors[i - 1], label=y_axis)
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.title(f"{title} Route {i - 1}")
            plt.legend()
            i += 1
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{title}.png'))
        plt.close()
        #plt.show()

    def plot_route_specific_tuple(self, evaluation_data, title):
        plt.figure(figsize=(10, 8))
        colors = ['b-', 'g-', 'r-', 'm-']
        i = 1
        for route in evaluation_data:
            plt.subplot(2, 2, i)
            plt.plot(route[0], -route[1], colors[i - 1], label=title)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f"{title} Route {i - 1}")
            plt.legend()
            i += 1
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{title}.png'))
        plt.close()
        #plt.show()

    def plot_heatmap(self, df):
        '''
        plt.figure(figsize=(10, 6),dpi=300)
        sns.scatterplot(x=df['gps_y'], y=df['gps_x'], hue=df['speed'], palette='Spectral', s=50)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"Speed Heatmap Along GPS Trajectory")
        plt.grid(True)
        plt.show()
        '''
        fig, ax = plt.subplots(figsize=(10, 8), dpi=500)

        # Scatter plot GPS coordinates colored by speed
        sc = ax.scatter(df['gps_x'], -1 * df['gps_y'], c=df['speed'], cmap='viridis', marker='o', edgecolor='face',
                        s=10)

        # Add color bar for speed
        cbar = plt.colorbar(sc)
        cbar.set_label('Speed (m/s)')

        # Set plot labels and title
        ax.set_xlabel('GPS X Coordinate')
        ax.set_ylabel('GPS Y Coordinate')
        ax.set_title('GPS Trajectory with Speed Overlay')

        # Show the plot
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'Speed_GPS_HeatMap.png'))
        plt.close()
        #plt.show()

    def plot_heatmap_individual(self, dfs):
        i = 0
        for df in dfs:
            fig, ax = plt.subplots(figsize=(20, 16), dpi=300)

            # Scatter plot GPS coordinates colored by speed
            sc = ax.scatter(df['gps_x'], -1 * df['gps_y'], c=df['speed'], cmap='viridis', marker='o', edgecolor='face',
                            s=10)

            # Add color bar for speed
            cbar = plt.colorbar(sc)
            cbar.set_label('Speed (m/s)')

            # Set plot labels and title
            ax.set_xlabel('GPS X Coordinate')
            ax.set_ylabel('GPS Y Coordinate')
            ax.set_title(f'GPS Trajectory with Speed Overlay for Route {i}')

            # Show the plot
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f'Speed_GPS_HeatMap_Route_{i}.png'))
            plt.close()
            #plt.show()
            i += 1

    def parse_evaluation_results(self, _town_maps, xml_file):
        _reference_coord = (-518.496, 398.342)
        _scale = (708/940, 627/844)
        infraction_to_symbol = {"collisions_layout": ("#ff0000", "."),
                                "collisions_pedestrian": ("#00ff00", "."),
                                "collisions_vehicle": ("#0000ff", "."),
                                "outside_route_lanes": ("#00ffff", "."),
                                "red_light": ("#ffff00", "."),
                                "route_dev": ("#ff00ff", "."),
                                "route_timeout": ("#ffffff", "."),
                                "stop_infraction": ("#777777", "."),
                                "vehicle_blocked": ("#000000", ".")}
        town = "Town04"
        def getPixel(coord):
            x, y = coord
            pix_x = int((x - _reference_coord[0]) * _scale[0])
            pix_y = int(-(-y - _reference_coord[1]) * _scale[1])
            return pix_x, pix_y


        def plotPixel(coord, town_name, town_img, color):
            pix_x, pix_y = getPixel(coord)

            length = 6
            width = 3
            town_img[pix_y - length:pix_y + (length + 1), pix_x - width:pix_x + (width + 1)] = color
            town_img[pix_y - width:pix_y + (width + 1), pix_x - length:pix_x + (length + 1)] = color

            return town_img

        def create_legend():
            symbols = [lines.Line2D([], [], color=col, marker=mark, markersize=15)
                       for _, (col, mark) in infraction_to_symbol.items()]
            names = [infraction for infraction, _ in infraction_to_symbol.items()]

            figlegend = plt.figure(figsize=(3, int(0.34 * len(names))))
            figlegend.legend(handles=symbols, labels=names)
            figlegend.savefig(os.path.join(self.output_dir, 'legend.png'))

        def hex_to_list(hex_str):
            hex_to_dec = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
                          "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
                          "a": 10, "b": 11, "c": 12, "d": 13,
                          "e": 14, "f": 15}

            num1 = 16 * hex_to_dec[hex_str[1]] + hex_to_dec[hex_str[2]]
            num2 = 16 * hex_to_dec[hex_str[3]] + hex_to_dec[hex_str[4]]
            num3 = 16 * hex_to_dec[hex_str[5]] + hex_to_dec[hex_str[6]]

            return [num1, num2, num3]

        def get_infraction_coords(infraction_description):
            combined = re.findall('\(x=.*\)', infraction_description)
            if len(combined) > 0:
                coords_str = combined[0][1:-1].split(", ")
                coords = [float(coord[2:]) for coord in coords_str]
            else:
                coords = ["-", "-", "-"]

            return coords

        root = ET.parse(xml_file).getroot()

        # build route matching dict
        route_matching = {}
        if ('weather' in [elem.tag for elem in root.iter()]):
            for route, weather_daytime in zip(root.iter('route'), root.iter('weather')):
                combined = re.findall('[A-Z][^A-Z]*', weather_daytime.attrib["id"])
                weather = "".join(combined[:-1])
                daytime = combined[-1]
                route_matching[route.attrib["id"]] = {'town': route.attrib["town"],
                                                      'weather': weather,
                                                      "daytime": daytime}
        else:
            for route in root.iter('route'):
                route_matching[route.attrib["id"]] = {'town': route.attrib["town"],
                                                      'weather': "Clear",
                                                      "daytime": "Noon"}

        # _, _, filenames = next(walk(args.results))
        filename = self.eval_results_file

        # lists to aggregate multiple json files
        route_evaluation = []
        total_score_labels = []
        total_score_values = []

        total_km_driven = 0.0
        total_infractions = {}
        total_infractions_per_km = {}

        abort = False
        # aggregate files
        with open(filename) as json_file:
            evaluation_data = json.load(json_file)

            if (len(total_infractions) == 0):
                for infraction_name in evaluation_data['_checkpoint']['global_record']['infractions']:
                    total_infractions[infraction_name] = 0

            for record in evaluation_data['_checkpoint']['records']:
                if (record['scores']['score_route'] <= 0.00000000001):
                    print("Warning: There is a route where the agent did not start to drive." + " Route ID: " +
                          record['route_id'], file=sys.stderr)
                if (record['status'] == "Failed - Agent couldn\'t be set up"):
                    print(
                        "Error: There is at least one route where the agent could not be set up. Aborting." + " Route ID: " +
                        record['route_id'], file=sys.stderr)
                    abort = True
                if (record['status'] == "Failed"):
                    print("Error: There is at least one route that failed. Aborting." + " Route ID: " + record[
                        'route_id'], file=sys.stderr)
                    abort = True
                if (record['status'] == "Failed - Simulation crashed"):
                    print(
                        "Error: There is at least one route where the simulation crashed. Aborting." + " Route ID: " +
                        record['route_id'], file=sys.stderr)
                    abort = True

                percentage_of_route_completed = record['scores']['score_route'] / 100.0
                route_length_km = record['meta']['route_length'] / 1000.0
                driven_km = percentage_of_route_completed * route_length_km
                total_km_driven += driven_km

                for infraction_name in evaluation_data['_checkpoint']['global_record']['infractions']:
                    if (infraction_name == 'outside_route_lanes'):
                        if (len(record['infractions'][infraction_name]) > 0):
                            meters_off_road = re.findall("\d+\.\d+", record['infractions'][infraction_name][0])[0]
                            km_off_road = float(meters_off_road) / 1000.0
                            total_infractions[infraction_name] += km_off_road
                    else:
                        num_infraction = len(record['infractions'][infraction_name])
                        total_infractions[infraction_name] += num_infraction

            eval_data = evaluation_data['_checkpoint']['records']
            total_scores = evaluation_data["values"]
            route_evaluation += eval_data

            total_score_labels = evaluation_data["labels"]
            total_score_values += [[float(score) * len(eval_data) for score in total_scores]]

        for key in total_infractions:
            total_infractions_per_km[key] = total_infractions[key] / total_km_driven
            if (key == 'outside_route_lanes'):
                total_infractions_per_km[key] = total_infractions_per_km[
                                                    key] * 100.0  # Since this infraction is a percentage, we put it in rage [0.0, 100.0]

        total_score_values = np.array(total_score_values)

        if ((len(route_evaluation) % len(route_matching) != 0)):
            print("Error: The number of completed routes (" + str(
                len(route_evaluation)) + ") is not a multiple of the total routes (" + str(
                len(route_matching)) + "). Check if there are missing results. Aborting.", file=sys.stderr)
            abort = True

        if (abort == True):
            exit()

        total_score_values = total_score_values.sum(axis=0) / len(route_evaluation)

        for idx, value in enumerate(total_score_labels):
            if (value == 'Collisions with pedestrians'):
                total_score_values[idx] = total_infractions_per_km['collisions_pedestrian']
            elif (value == 'Collisions with vehicles'):
                total_score_values[idx] = total_infractions_per_km['collisions_vehicle']
            elif (value == 'Collisions with layout'):
                total_score_values[idx] = total_infractions_per_km['collisions_layout']
            elif (value == 'Red lights infractions'):
                total_score_values[idx] = total_infractions_per_km['red_light']
            elif (value == 'Stop sign infractions'):
                total_score_values[idx] = total_infractions_per_km['stop_infraction']
            elif (value == 'Off-road infractions'):
                total_score_values[idx] = total_infractions_per_km['outside_route_lanes']
            elif (value == 'Route deviations'):
                total_score_values[idx] = total_infractions_per_km['route_dev']
            elif (value == 'Route timeouts'):
                total_score_values[idx] = total_infractions_per_km['route_timeout']
            elif (value == 'Agent blocked'):
                total_score_values[idx] = total_infractions_per_km['vehicle_blocked']

        # dict to extract unique identity of route in case of repetitions
        route_to_id = {}
        for route in route_evaluation:
            route_to_id[route["route_id"]] = ''.join(i for i in route["route_id"] if i.isdigit())

        # build table of relevant information
        total_score_info = [{"label": label, "value": value} for label, value in
                            zip(total_score_labels, total_score_values)]
        route_scenarios = [{"route": route["route_id"],
                            "town": route_matching[route_to_id[route["route_id"]]]["town"],
                            "weather": route_matching[route_to_id[route["route_id"]]]["weather"],
                            "daytime": route_matching[route_to_id[route["route_id"]]]["daytime"],
                            "duration": route["meta"]["duration_game"],
                            "length": route["meta"]["route_length"],
                            "score": route["scores"]["score_composed"],
                            "completion": route["scores"]["score_route"],
                            "status": route["status"],
                            "infractions": [(key,
                                             len(item),
                                             [get_infraction_coords(description) for description in item])
                                            for key, item in route["infractions"].items()]}
                           for route in route_evaluation]

        # compute aggregated statistics and table for each filter
        filters = ["route", "town", "weather", "daytime", "status"]
        evaluation_filtered = {}

        for filter in filters:
            subcategories = np.unique(np.array([scenario[filter] for scenario in route_scenarios]))
            route_scenarios_per_subcategory = {}
            evaluation_per_subcategory = {}
            for subcategory in subcategories:
                route_scenarios_per_subcategory[subcategory] = []
                evaluation_per_subcategory[subcategory] = {}
            for scenario in route_scenarios:
                route_scenarios_per_subcategory[scenario[filter]].append(scenario)
            for subcategory in subcategories:
                scores = np.array([scenario["score"] for scenario in route_scenarios_per_subcategory[subcategory]])
                completions = np.array(
                    [scenario["completion"] for scenario in route_scenarios_per_subcategory[subcategory]])
                durations = np.array(
                    [scenario["duration"] for scenario in route_scenarios_per_subcategory[subcategory]])
                lengths = np.array([scenario["length"] for scenario in route_scenarios_per_subcategory[subcategory]])

                infractions = np.array([[infraction[1] for infraction in scenario["infractions"]]
                                        for scenario in route_scenarios_per_subcategory[subcategory]])

                scores_combined = (scores.mean(), scores.std())
                completions_combined = (completions.mean(), completions.std())
                durations_combined = (durations.mean(), durations.std())
                lengths_combined = (lengths.mean(), lengths.std())
                infractions_combined = [(mean, std) for mean, std in
                                        zip(infractions.mean(axis=0), infractions.std(axis=0))]

                evaluation_per_subcategory[subcategory] = {"score": scores_combined,
                                                           "completion": completions_combined,
                                                           "duration": durations_combined,
                                                           "length": lengths_combined,
                                                           "infractions": infractions_combined}
            evaluation_filtered[filter] = evaluation_per_subcategory

        # write output csv file
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        f = open(os.path.join(self.output_dir, 'results.csv'), 'w')  # Make file object first
        csv_writer_object = csv.writer(f)  # Make csv writer object
        # writerow writes one row of data given as list object
        for info in total_score_info:
            # print([info[key] for key in info.keys()])
            csv_writer_object.writerow([item for _, item in info.items()])
        csv_writer_object.writerow([""])

        for filter in filters:
            infractions_types = []
            for infraction in route_scenarios[0]["infractions"]:
                infractions_types.append(infraction[0] + " mean")
                infractions_types.append(infraction[0] + " std")

            # route aggregation table has additional columns
            if filter == "route":
                csv_writer_object.writerow(
                    [filter, "town", "weather", "daytime", "score mean", "score std", "completion mean",
                     "completion std", "duration mean", "duration std", "length mean", "length std"] +
                    infractions_types)
            else:
                csv_writer_object.writerow(
                    [filter, "score mean", "score std", "completion mean", "completion std", "duration mean",
                     "duration std", "length mean", "length std"] +
                    infractions_types)

            sorted_keys = sorted(evaluation_filtered[filter].keys(),
                                 key=lambda x: int(x.split('_')[-1]) if len(x.split('_')) >= 2 else -1)
            # sorted_keys = sorted(evaluation_filtered[filter].keys(), lambda x: -1)

            # for key,item in evaluation_filtered[filter].items():
            for key in sorted_keys:
                item = evaluation_filtered[filter][key]
                infractions_output = []
                for infraction in item["infractions"]:
                    infractions_output.append(infraction[0])
                    infractions_output.append(infraction[1])
                if filter == "route":
                    csv_writer_object.writerow([key,
                                                route_matching[route_to_id[key]]["town"],
                                                route_matching[route_to_id[key]]["weather"],
                                                route_matching[route_to_id[key]]["daytime"],
                                                item["score"][0], item["score"][1],
                                                item["completion"][0], item["completion"][1],
                                                item["duration"][0], item["duration"][1],
                                                item["length"][0], item["length"][1]] +
                                               infractions_output)
                else:
                    csv_writer_object.writerow([key,
                                                item["score"][0], item["score"][1],
                                                item["completion"][0], item["completion"][1],
                                                item["duration"][0], item["duration"][1],
                                                item["length"][0], item["length"][1]] +
                                               infractions_output)
            csv_writer_object.writerow([""])

        csv_writer_object.writerow(["town", "weather", "daylight", "infraction type", "x", "y", "z"])
        # writerow writes one row of data given as list object
        for scenario in route_scenarios:
            # print([scenario[key] for key in scenario.keys() if key!="town"])
            for infraction in scenario["infractions"]:
                for coord in infraction[2]:
                    if type(coord[0]) != str:
                        csv_writer_object.writerow([scenario["town"], scenario["weather"], scenario["daytime"],
                                                    infraction[0]] + coord)
        csv_writer_object.writerow([""])
        f.close()

        # load town maps for plotting infractions
        #town_maps = {}
        #town_maps[town] = np.array(Image.open(os.path.join(town_maps, town + '.png')))[:, :, :3]
        town_map = np.array(Image.open(os.path.join(_town_maps, town + ".png")))[:, :, :3]

        create_legend()

        for scenario in route_scenarios:
            for infraction in scenario["infractions"]:
                for coord in infraction[2]:
                    if type(coord[0]) != str:
                        x = coord[0]
                        y = coord[1]

                        #town_name = scenario["town"]

                        hex_str, _ = infraction_to_symbol[infraction[0]]
                        color = hex_to_list(hex_str)
                        # plot infractions
                        town_map = plotPixel((x, y), town, town_map, color)

        tmap = Image.fromarray(town_map)
        tmap.save(os.path.join(self.output_dir, town + '.png'))

def extract_data(converted_dataframe, param):
    return converted_dataframe.groupby('route')[param].apply(list).to_list()


def extract_coordinate_data(converted_dataframe, param1, param2):
    return converted_dataframe.groupby('route').apply(
        lambda g: [np.array(g[param1].tolist()), np.array(g[param2].tolist())]).tolist()


def transform_dataframe(converted_dataframe):
    return [group for _, group in converted_dataframe.groupby('route')]


def extract_route_data(route_data):
    return [np.array(list(map(list, zip(*x)))) for x in route_data]

# Usage Example:
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output = os.path.join("../evaluation_folder/results/vis_outputs", f"run_{timestamp}")
base_dirs = ['final_evaluation/org_seed_1/org_seed_1_eval_1_debug',
             'final_evaluation/org_seed_1/org_seed_1_eval_2_debug',
             'final_evaluation/org_seed_1/org_seed_1_eval_3_debug',
             'final_evaluation/train_num_14_mod_14/train_num_14_mod_14_eval_1_debug',
             'final_evaluation/train_num_14_mod_14/train_num_14_mod_14_eval_2_debug',
             'final_evaluation/train_num_14_mod_14/train_num_14_mod_14_eval_3_debug'
             ]
base_dirs = ['results/train_num_14_mod_20/train_num_14_mod_20_eval_3',
             'results/train_num_14_mod_20/train_num_14_mod_20_eval_2',
             'results/train_num_14_mod_20/train_num_14_mod_20_eval_1']

for base_dir in base_dirs:
    processor = CarlaEvaluationProcessor(base_dir, output)
    processor.create_video(3)
#exit()

xml_file_path = '../evaluation_folder/evaluation_route.xml'
routes_data = parse_xml(xml_file_path)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
stats = {}

exit()
def average_n(df, n):
    # Group by the 'route' column to ensure values from different routes are not averaged together
    grouped = df.groupby('route')

    # Initialize an empty list to store the averaged dataframes
    averaged_dfs = []

    # Loop through each group (i.e., each route)
    for route_value, group in grouped:
        # Select columns with integer values and keep the 'route' column
        int_columns = group.select_dtypes(include='float')
        # Add the 'route' column back to the integer columns
        int_columns['route'] = group['route']

        # Calculate the mean for every n rows, without averaging the 'route' column
        averaged_group = int_columns.groupby(int_columns.index // n).mean()

        # Set the route value correctly (since it gets averaged, we need to reassign it)
        averaged_group['route'] = route_value

        # Append the processed group to the list
        averaged_dfs.append(averaged_group)

    # Concatenate all the averaged groups back into a single DataFrame
    result_df = pd.concat(averaged_dfs).reset_index(drop=True)

    return result_df
'''
base_dir1 = 'results/train_num_14_mod_14/train_num_14_mod_14_eval_3'
base_dir2 = 'results/org_seed_1/org_seed_1_eval_2'

processor1 = CarlaEvaluationProcessor(base_dir1, output)
processor2 = CarlaEvaluationProcessor(base_dir2, output)
evaluation_data1, evalution_list_data = processor1.load_evaluation_time_data()
evaluation_data2, evalution_list_data = processor2.load_evaluation_time_data()

converted_dataframe1 = processor1.generate_converted_waypoints(evaluation_data1, routes_data)
converted_dataframe2 = processor2.generate_converted_waypoints(evaluation_data2, routes_data)
#processor.plot_map(converted_dataframe, routes_data, "evaluation_plot1.png")
converted_dataframe1 = average_n(converted_dataframe1, 8)
converted_dataframe2 = average_n(converted_dataframe2, 8)
speed1 = extract_data(converted_dataframe1, "speed")
speed2 = extract_data(converted_dataframe2, "speed")
avg1 = 0
counter = 0
for s in speed1:
    for p in s:
        avg1 += p
        counter += 1
avg1 /= counter
print(avg1*3.6)
avg1 = 0
counter = 0
for s in speed2:
    for p in s:
        avg1 += p
        counter += 1
avg1 /= counter
print(avg1*3.6)
#for i in range(len(speed1[3]),len(speed2[3])):
 #   speed1[3].append(0)

plt.figure(figsize=(10, 6), dpi=300)
index1 = range(len(speed1[3]))
index2 = range(len(speed2[3]))
plt.subplot(2, 1, 1)
plt.plot(index1, speed1[3], 'r-', label="Speed Fine-Tuned Model")
plt.xlabel("Timestep")
plt.xlim(-149.0, 3129.0)
plt.ylabel("Speed (m/s)")
plt.title("Speed Fine-Tuned Model")
#plt.legend()
#i += 1
#plt.tight_layout()


plt.subplot(2, 1, 2)
plt.plot(index2, speed2[3], 'b-', label="Speed Original Model")
plt.savefig(os.path.join("test", "speed_time.png"))
plt.xlabel("Timestep")
plt.ylabel("Speed (m/s)")
plt.title("Speed Original Model")
plt.tight_layout()
plt.savefig(os.path.join("test", "speed_time.png"))
plt.close()
#processor.plot_route_specific_index(speed, x_axis="Timestep", y_axis="Speed (m/s)", title='Speed Over Time')

exit()
'''

for base_dir in base_dirs:
    processor = CarlaEvaluationProcessor(base_dir, output)
    evaluation_data, evalution_list_data = processor.load_evaluation_time_data()
    route_counter = 0
    dir = os.path.dirname(base_dir)
    for route in evalution_list_data:
        continue
        #print(route)
        #print(f"Route {route_counter}: {route[route['speed'] >= 4.0].index[0]}")
        #print(f"Route {route_counter}: {route.index[route['speed'] >= 4.0][0]}")
        inertia = route.index[route['speed'] >= 4.0][0]
        if not dir in stats:
            stats[dir] = 0
        if not f"{dir}/route{route_counter}" in stats:
            stats[f"{dir}/route{route_counter}"] = 0
        stats[f"{dir}/route{route_counter}"] += inertia
        stats[dir] += inertia
        route_counter += 1
        #processor.create_video(route=route_counter)

    converted_dataframe = processor.generate_converted_waypoints(evaluation_data, routes_data)
    processor.plot_map(converted_dataframe, routes_data, "evaluation_plot1.png")
    converted_dataframe = average_n(converted_dataframe, 8)
    compass = extract_data(converted_dataframe, "compass")
    speed = extract_data(converted_dataframe, "speed")
    position = extract_coordinate_data(converted_dataframe, "position_x", "position_y")
    world = extract_coordinate_data(converted_dataframe, "world_x", "world_y")
    gps = extract_coordinate_data(converted_dataframe, "gps_x", "gps_y")
    target = extract_coordinate_data(converted_dataframe, "target_point_x", "target_point_y")
    target_global = extract_coordinate_data(converted_dataframe, "global_target_x", "global_target_y")
    routes_locations = extract_route_data(routes_data)

    processor.plot_route_specific_index(compass, x_axis="Timestep", y_axis="Compass Value",
                                    title='Compass Values Over Time')
    processor.plot_route_specific_index(speed, x_axis="Timestep", y_axis="Speed (m/s)", title='Speed Over Time')
    processor.plot_route_specific_tuple(position, title='Vehicle Position')
    processor.plot_route_specific_tuple(gps, title='GPS Trajectory')
    processor.plot_route_specific_tuple(target, title='Target Point')
    processor.plot_route_specific_tuple(target_global, title='Target Point Global')
    processor.plot_route_specific_tuple(world, title='World Position')
    processor.plot_route_specific_tuple(routes_locations, title='Waypoints')
    converted_dataframe = average_n(converted_dataframe, 8)
    processor.plot_heatmap(converted_dataframe)
    processor.plot_heatmap_individual(transform_dataframe(converted_dataframe))
    #print(len(speed))


for x in stats.items():
    if "route" in x[0]:
        print(x[0], x[1]/3)
    else:
        print(x[0], x[1]/12)

#xml_file = 'evaluation_route.xml'
#maps = "../leaderboard/data/town_maps_xodr"
#processor.parse_evaluation_results(maps, xml_file)

# 1. Create video for route 0
#processor.create_video(route=3)
#exit()
# 2. Load and process speed data
exit()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
evaluation_data = processor.load_evaluation_time_data()

# 3. Parse XML file (assuming XML file path is provided)
xml_file = '../evaluation_folder/evaluation_route.xml'
routes_data = processor.parse_xml(xml_file)


# 4. Convert the x,y coordinates from BEV plane to image plane and include the transformed world coordinates
converted_dataframe = processor.generate_converted_waypoints(evaluation_data, routes_data)

processor.plot_map(converted_dataframe, routes_data, "evaluation_plot1.png")
exit()
routes_data = [routes_data[3][80:120]]
converted_dataframe = converted_dataframe[converted_dataframe['route'] == 'route3']
converted_dataframe = converted_dataframe[10500:16300]
print(len(converted_dataframe))
#print(converted_dataframe.head())
# 5. Plot data
processor.plot_map(converted_dataframe, routes_data, "evaluation_plot1.png")
processor.plot_map(converted_dataframe, routes_data, "evaluation_plot2.png")
processor.plot_map(converted_dataframe, routes_data, "evaluation_plot3.png")
processor.plot_map(converted_dataframe, routes_data, "evaluation_plot4.png")
exit()




compass = extract_data(converted_dataframe, "compass")
speed = extract_data(converted_dataframe, "speed")
position = extract_coordinate_data(converted_dataframe, "position_x", "position_y")
world = extract_coordinate_data(converted_dataframe, "world_x", "world_y")
gps = extract_coordinate_data(converted_dataframe, "gps_x", "gps_y")
target = extract_coordinate_data(converted_dataframe, "target_point_x", "target_point_y")
target_global = extract_coordinate_data(converted_dataframe, "global_target_x", "global_target_y")
routes_locations = extract_route_data(routes_data)
processor.plot_route_specific_index(compass, x_axis="Timestep", y_axis="Compass Value",
                                    title='Compass Values Over Time')
processor.plot_route_specific_index(speed, x_axis="Timestep", y_axis="Speed (m/s)", title='Speed Over Time')
processor.plot_route_specific_tuple(position, title='Vehicle Position')
processor.plot_route_specific_tuple(gps, title='GPS Trajectory')
processor.plot_route_specific_tuple(target, title='Target Point')
processor.plot_route_specific_tuple(target_global, title='Target Point Global')
processor.plot_route_specific_tuple(world, title='World Position')
processor.plot_route_specific_tuple(routes_locations, title='Waypoints')
processor.plot_heatmap(converted_dataframe)
processor.plot_heatmap_individual(transform_dataframe(converted_dataframe))
