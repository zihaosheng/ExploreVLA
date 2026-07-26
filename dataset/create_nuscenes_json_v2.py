import argparse
import os
import json
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

TARGET_PERIOD = 0.5  # 2Hz = 0.5s
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def collect_cam_front_frames(nusc, scene):
    """Return a list of (sample_data_token, filename) sampled at 2Hz."""
    frames = []

    # first CAM_FRONT sample_data
    sample_token = scene["first_sample_token"]
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["CAM_FRONT"]
    sd = nusc.get("sample_data", sd_token)

    frames.append((sample_token, sd["filename"]))

    cumul_delta = 0.0

    while sample["next"] != "":
        next_sample = nusc.get("sample", sample["next"])
        next_sd_token = next_sample["data"]["CAM_FRONT"]
        next_sd = nusc.get("sample_data", next_sd_token)
        delta = (next_sample["timestamp"] - sample["timestamp"]) / 1e6
        cumul_delta += delta

        # if cumul_delta >= TARGET_PERIOD:
        #     frames.append((next_sample["token"], next_sd["filename"]))
        #     cumul_delta -= TARGET_PERIOD

        frames.append((next_sample["token"], next_sd["filename"]))

        sample = next_sample

    return frames


def build_json_fast(nusc, split="train", output_dir=SCRIPT_DIR):
    result = {}
    scene_splits = create_splits_scenes()[split]
    output_json = f"nusc_2hz_{split}_v2.json"

    # 取 scene
    scenes = [s for s in nusc.scene if s["name"] in scene_splits]

    for scene in tqdm(scenes, desc="Processing scenes"):
        frames = collect_cam_front_frames(nusc, scene)

        # 为每一帧生成 prev, next
        # for i, (tok, fname) in enumerate(frames):
        for i in range(1, len(frames) - 1):
            tok, fname = frames[i]
            _, prev_fname = frames[i - 1]
            _, next_fname = frames[i + 1]
            result[tok] = {"prev": [prev_fname, fname], "next": [next_fname]}

    with open(os.path.join(output_dir, output_json), "w") as f:
        json.dump(result, f, indent=4)

    print("Saved:", output_json)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True, help="Path to the nuScenes dataset root")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes dataset version")
    parser.add_argument("--output-dir", default=SCRIPT_DIR, help="Directory to save the output json files")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    build_json_fast(nusc, split="train", output_dir=args.output_dir)
    build_json_fast(nusc, split="val", output_dir=args.output_dir)
