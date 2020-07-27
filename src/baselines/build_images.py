import argparse
import os
import json
from PIL import Image, ImageDraw
import glob
import time
import sys
sys.path.append('../../')
from config.global_config import GlobalConfig

config = GlobalConfig()
IMG_WIDTH, IMG_HEIGHT = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
EMOJI_HEIGHT, EMOJI_WIDTH = config.EMOJI_HEIGHT, config.EMOJI_WIDTH


def render_scene(scene, args):
    objects = scene['objects']

    img = Image.new('RGBA', (IMG_WIDTH, IMG_HEIGHT), color=(0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    emojis = [Image.open(fname).convert('RGBA') for fname in sorted(glob.glob('../../emojis/*.png'))]

    emoji_objs = [os.path.split(fname)[-1].split('.')[0] for fname in sorted(glob.glob('../../emojis/*.png'))]

    src_w, src_h = None, None
    for object in objects:
        emoji_name = '%s_%s' % (object['id'], object['name'])
        emoji_idx = emoji_objs.index(emoji_name)
        emoji_w, emoji_h = object['width'], object['height']
        px, py = object['pixel_coord'][0], object['pixel_coord'][1]
        img.paste(emojis[emoji_idx].resize((emoji_w, emoji_h)), [px - emoji_w // 2, py - emoji_h // 2],
                  emojis[emoji_idx].resize((emoji_w, emoji_h)))
        if 'is_source' in object.keys() and object['is_source']:
            src_w, src_h = emoji_w, emoji_h

    if scene['target_type'] == 'single':
        tpx, tpy = scene['target_pixel_coord'][0], scene['target_pixel_coord'][1]
        draw.rectangle([(int(tpx) - src_w // 2 - 4, int(tpy) - src_h // 2 - 4),
                        (int(tpx) + src_w // 2 + 4, int(tpy) + src_h // 2 + 4)], outline="green", fill=None)

    uniq_img_id = scene['unique_img_id']
    print('Image ID: {} | Instruction: {}'.format(uniq_img_id,scene['instruction']))
    image_path = os.path.join(args.image_dir, args.template, args.action, args.target,
                              'img_' + str(uniq_img_id) + '.png')
    img.convert('RGB').save(image_path)


def main(args):
    start_time = time.time()
    image_dir = os.path.join(args.image_dir, args.template, args.action, args.target)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    scene_path = os.path.join(args.scene_dir, args.template,
                              args.action + '_' +
                              args.target + '_' +
                              'scenes.json')
    with open(scene_path, 'r') as f:
        scenes = json.load(f)['scenes']
    for scene in scenes:
        render_scene(scene, args)

    print('*' * 50)
    print('Time required: {:.2f} s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(description='Question generation framework.')
    parser.add_argument('--template', default='SP_SO_REL_SOR', help='Template type.'),
    parser.add_argument('--action', default='pick', help='Action type. pick | pick_place')
    parser.add_argument('--target', default='single', type=str, help='Target type. single or multi.')
    parser.add_argument('--image_dir', default='../../data/images/', type=str, help='Directory to store images.')
    parser.add_argument('--scene_dir', default='../../data/scenes/', type=str, help='Directory of stored scene data.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
