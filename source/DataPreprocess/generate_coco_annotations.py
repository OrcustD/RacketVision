"""Generate COCO-format annotation files from native racket JSON annotations.

Produces two types of COCO files:
  - Detection: bboxes with sport-specific racket category IDs (+ empty ball category)
  - Pose: bboxes + 5 keypoints with a single 'racket' category

Usage:
    python generate_coco_annotations.py --data_root ../data
"""

import argparse
import json
import os
import glob


SPORT_TO_DET_CATEGORY = {
    'badminton': 2,
    'tabletennis': 3,
    'tennis': 4,
}

DET_CATEGORIES = [
    {'id': 1, 'name': 'ball', 'supercategory': 'object'},
    {'id': 2, 'name': 'badminton_racket', 'supercategory': 'racket'},
    {'id': 3, 'name': 'tabletennis_racket', 'supercategory': 'racket'},
    {'id': 4, 'name': 'tennis_racket', 'supercategory': 'racket'},
]

POSE_CATEGORIES = [{
    'id': 1,
    'name': 'racket',
    'supercategory': 'racket',
    'keypoints': ['top', 'bottom', 'handle', 'left', 'right'],
    'skeleton': [[1, 4], [1, 5], [2, 4], [2, 5], [2, 3], [1, 2], [4, 5]],
}]


def load_dataset_info(data_root):
    path = os.path.join(data_root, 'annotations', 'dataset_info.json')
    with open(path, 'r') as f:
        return json.load(f)


def build_coco(data_root, dataset_info, clip_indices, mode='det'):
    """Build a COCO-format dict for the given clip indices.

    Args:
        mode: 'det' for detection, 'pose' for keypoint estimation.
    """
    images = []
    annotations = []
    img_id = 0
    ann_id = 0

    for idx in clip_indices:
        clip = dataset_info['clips'][idx]
        sport = clip['sport']
        match_id = clip['match_id']
        round_id = clip['round_id']
        w = clip.get('width', 1920)
        h = clip.get('height', 1080)

        racket_dir = os.path.join(
            data_root, sport, 'all', match_id, 'racket', round_id)
        if not os.path.isdir(racket_dir):
            continue

        racket_files = sorted(glob.glob(os.path.join(racket_dir, '*.json')))
        for rfile in racket_files:
            frame_id = int(os.path.basename(rfile).split('.')[0])
            img_path = os.path.join(
                sport, 'all', match_id, 'frame', round_id,
                f'{frame_id:04d}.jpg')
            full_img_path = os.path.join(data_root, img_path)
            if not os.path.isfile(full_img_path):
                continue

            with open(rfile, 'r') as f:
                racket_data = json.load(f)

            has_valid_ann = False
            frame_anns = []

            for racket in racket_data:
                if 'bbox_xywh' not in racket:
                    continue
                bx, by, bw, bh = racket['bbox_xywh']
                area = bw * bh
                if area <= 0:
                    continue

                ann = {
                    'id': ann_id,
                    'image_id': img_id,
                    'bbox': [bx, by, bw, bh],
                    'area': area,
                    'iscrowd': 0,
                }

                if mode == 'det':
                    ann['category_id'] = SPORT_TO_DET_CATEGORY[sport]
                else:
                    ann['category_id'] = 1
                    kpts_raw = racket.get('keypoints', [])
                    keypoints = []
                    num_kp = 0
                    for i in range(5):
                        if i < len(kpts_raw) and len(kpts_raw[i]) >= 3:
                            kx, ky, kv = float(kpts_raw[i][0]), float(kpts_raw[i][1]), int(kpts_raw[i][2])
                            if kv > 0:
                                keypoints.extend([kx, ky, 2])
                                num_kp += 1
                            else:
                                keypoints.extend([0, 0, 0])
                        else:
                            keypoints.extend([0, 0, 0])
                    ann['keypoints'] = keypoints
                    ann['num_keypoints'] = num_kp
                    if num_kp < 3:
                        continue

                frame_anns.append(ann)
                ann_id += 1
                has_valid_ann = True

            if has_valid_ann:
                images.append({
                    'id': img_id,
                    'file_name': img_path.replace('\\', '/'),
                    'width': w,
                    'height': h,
                })
                annotations.extend(frame_anns)
                img_id += 1

    categories = DET_CATEGORIES if mode == 'det' else POSE_CATEGORIES
    return {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    dataset_info = load_dataset_info(data_root)

    splits = dataset_info['splits']
    out_dir = os.path.join(data_root, 'info')
    os.makedirs(out_dir, exist_ok=True)

    for split_name, clip_indices in splits.items():
        for mode in ['det', 'pose']:
            coco = build_coco(data_root, dataset_info, clip_indices, mode=mode)
            fname = f'{split_name}_{mode}_coco.json'
            out_path = os.path.join(out_dir, fname)
            with open(out_path, 'w') as f:
                json.dump(coco, f, indent=2)
            n_img = len(coco['images'])
            n_ann = len(coco['annotations'])
            print(f'[{split_name}/{mode}] {n_img} images, {n_ann} annotations -> {out_path}')


if __name__ == '__main__':
    main()
