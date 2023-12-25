import os
import argparse
import logging
import numpy as np
import pandas as pd
import json
from PIL import Image
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def read_images(image_root):
    image_files = sorted(os.listdir(image_root),
                         key=lambda x: int(x.split('_')[-2]))
    images = [np.array(Image.open(os.path.join(image_root, file)))
              for file in image_files]
    return images


def read_csv_data(csv_path):
    return pd.read_csv(csv_path)


def read_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['natural_language_description']


def main(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    for subdir in input_folder.iterdir():
        if subdir.is_dir():
            # Define paths for different image types and data files
            rgb_path = subdir / 'rgb'
            depth_path = subdir / 'depth'
            cam_01_path = subdir / 'cam_01'
            cam_02_path = subdir / 'cam_02'
            cam_fisheye_path = subdir / 'cam_fisheye'
            csv_path = subdir / 'result.csv'
            json_path = subdir / 'info.json'

            # Read data
            rgb_images = read_images(rgb_path) if rgb_path.exists() else []
            depth_images = read_images(
                depth_path) if depth_path.exists() else []
            cam_01_images = read_images(
                cam_01_path) if cam_01_path.exists() else []
            cam_02_images = read_images(
                cam_02_path) if cam_02_path.exists() else []
            cam_fisheye_images = read_images(
                cam_fisheye_path) if cam_fisheye_path.exists() else []
            csv_data = read_csv_data(
                csv_path) if csv_path.exists() else pd.DataFrame()
            language_instruction = read_json_data(
                json_path) if json_path.exists() else ""

            # Construct episodes
            episode = []
            for i in range(len(csv_data)):
                episode.append({
                    'image': rgb_images[i].astype(np.uint8) if i < len(rgb_images) else None,
                    'depth': (depth_images[i].astype(np.uint16)).reshape(720, 1280, 1) if i < len(depth_images) else None,
                    'image_left': cam_01_images[i].astype(np.uint8) if i < len(cam_01_images) else None,
                    'image_right': cam_02_images[i].astype(np.uint8) if i < len(cam_02_images) else None,
                    'image_fisheye': cam_fisheye_images[i].astype(np.uint8) if i < len(cam_fisheye_images) else None,
                    'end_effector_pose': csv_data.iloc[i][['camera_link_position_x', 'camera_link_position_y', 'camera_link_position_z', 'camera_link_orientation_x', 'camera_link_orientation_y', 'camera_link_orientation_z', 'camera_link_orientation_w']].to_numpy(dtype=np.float32),
                    'action': csv_data.iloc[i][['gripper_closed', 'ee_command_position_x', 'ee_command_position_y', 'ee_command_position_z', 'ee_command_rotation_x', 'ee_command_rotation_y', 'ee_command_rotation_z']].to_numpy(dtype=np.float32),
                    'language_instruction': language_instruction
                })

            # Save episode
            np.save(output_folder / f"episode_{subdir.stem}.npy", episode)
            logging.info(f"Saved episode_{subdir.stem}.npy to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and save episodes from folders.")
    parser.add_argument("--input_folder", required=True,
                        help="Input folder containing subfolders.")
    parser.add_argument("--output_folder", required=True,
                        help="Output folder to save episodes.")
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
    logging.info("Done!")
