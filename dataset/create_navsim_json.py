import argparse
import json
import os
from omegaconf import OmegaConf
from navsim.common.dataclasses import SensorConfig
from hydra.utils import instantiate
from pathlib import Path
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_FILTER_ROOT = os.path.join(
    REPO_ROOT, "navsim", "planning", "script", "config", "common", "train_test_split", "scene_filter"
)


def build_json_fast(scene_loader, nuplan_img_logs_root, split="train", output_dir=SCRIPT_DIR):
    result = {}
    output_json = f"navsim_2hz_{split}.json"

    # 取 scene
    scene_tokens = set([t.split('.')[0] for t in os.listdir(os.path.join(nuplan_img_logs_root, split))])

    for scene_token in tqdm(scene_tokens, desc="Processing scenes"):
        assert len(scene_loader.scene_frames_dicts[scene_token]) == 14, \
                f"scene_token={scene_token}, len={len(scene_loader.scene_frames_dicts[scene_token])}"
        prev_fname = scene_loader.scene_frames_dicts[scene_token][2]['cams']['CAM_F0']['data_path']
        curr_fname = scene_loader.scene_frames_dicts[scene_token][3]['cams']['CAM_F0']['data_path']
        next_fname = scene_loader.scene_frames_dicts[scene_token][4]['cams']['CAM_F0']['data_path']

        result[scene_token] = {"prev": [prev_fname, curr_fname], "next": [next_fname]}

    with open(os.path.join(output_dir, output_json), "w") as f:
        json.dump(result, f, indent=4)

    print("Saved:", output_json)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--navsim-root", required=True, help="Path to the NAVSIM dataset root")
    parser.add_argument("--nuplan-img-logs-root", required=True,
                         help="Path to the directory containing per-split nuPlan image log folders")
    parser.add_argument("--cache-path", required=True, help="Path to the NAVSIM metric cache")
    parser.add_argument("--filter-root", default=DEFAULT_FILTER_ROOT,
                         help="Path to the NAVSIM scene_filter config directory")
    parser.add_argument("--output-dir", default=SCRIPT_DIR, help="Directory to save the output json files")
    args = parser.parse_args()

    for test in [True, False]:
        splits = ['test'] if test else ['train', 'val']
        for split in splits:
            log_name = 'test' if test else 'trainval'
            filter_name = 'navtest' if test else 'navtrain'
            logs_data = os.path.join(args.navsim_root, 'navsim_logs', log_name)
            scene_filter = instantiate(OmegaConf.load(os.path.join(args.filter_root, f'{filter_name}.yaml')))
            metric_cache_loader = MetricCacheLoader(Path(args.cache_path))
            # scene_filter.tokens = metric_cache_loader.tokens
            scene_loader = SceneLoader(
                sensor_blobs_path=Path(args.navsim_root, 'sensor_blobs', log_name),
                data_path=Path(logs_data),
                scene_filter=scene_filter,
                sensor_config=SensorConfig.build_front_sensors(),
            )
            build_json_fast(scene_loader, args.nuplan_img_logs_root, split=split, output_dir=args.output_dir)
