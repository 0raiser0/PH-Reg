import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse

from mmengine.config import Config
from mmengine.runner import Runner

import custom_datasets  # This and the following should be loaded here because of mmseg module registration
import segmentor
import vishook


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation with MMSeg')
    parser.add_argument('--config', default='./configs/cfg_voc21.py')
    parser.add_argument('--backbone', default='')
    parser.add_argument('--pamr', default='off')
    parser.add_argument('--work-dir', default='./logs')
    parser.add_argument('--show-dir', default='', help='directory to save visualization images')
    args = parser.parse_args()
    return args


def visualization_hook(cfg, show_dir):
    if show_dir == '':
        cfg.default_hooks.pop('visualization', None)
        return
    if 'visualization' not in cfg.default_hooks:
        raise RuntimeError('VisualizationHook must be included in default_hooks, see base_config.py')
    else:
        hook = cfg.default_hooks['visualization']
        hook['draw'] = True
        visualizer = cfg.visualizer
        visualizer['save_dir'] = show_dir
        cfg.model['pamr_steps'] = 50
        cfg.model['pamr_stride'] = [1, 2, 4, 8, 12, 24]
        
def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        # if args.show:
        #     visualization_hook['show'] = True
        #     visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def safe_set_arg(cfg, arg, name, func=lambda x: x):
    if arg != '':
        cfg.model[name] = func(arg)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    safe_set_arg(cfg, args.backbone, 'clip_path')
    if args.pamr == 'off':
        cfg.model['pamr_steps'] = 0
    elif args.pamr == 'on':
        cfg.model['pamr_steps'] = 10
    # visualization_hook(cfg, args.show_dir)
    if args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
