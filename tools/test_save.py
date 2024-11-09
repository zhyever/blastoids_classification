import os
import os.path as osp
import argparse
import torch
import time
from torch.utils.data import DataLoader
from mmengine.utils import mkdir_or_exist
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger

from estimator.utils import RunnerInfo, setup_env, log_env, fix_random_seed
from estimator.models.builder import build_model
from estimator.datasets.builder import build_dataset
from estimator.tester import Tester
from mmengine import print_log

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--test_type',
        type=str,
        default='normal',
        help='evaluation type')
    parser.add_argument(
        '--tag',
        type=str,
        default='comm',
        help='tag name used in evaluation logging file')
    parser.add_argument(
        '--ckp_path',
        type=str,
        help='ckp_path')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='show prediction')
    parser.add_argument(
        '--test-day',
        type=str,
        default=None,
        help='test_day')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--save-name',
        type=str,
        default=None)
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    mkdir_or_exist(cfg.work_dir)
    cfg.ckp_path = args.ckp_path
    
    # fix seed
    seed = cfg.get('seed', 5621)
    fix_random_seed(seed)
    
    # start dist training
    if cfg.launcher == 'none':
        distributed = False
        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
        rank = 0
        world_size = 1
        env_cfg = cfg.get('env_cfg')
    else:
        distributed = True
        env_cfg = cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl')))
        rank, world_size, timestamp = setup_env(env_cfg, distributed, cfg.launcher)
    
    # build dataloader
    if args.test_type == 'consistency':
        dataset = build_dataset(cfg.val_consistency_dataloader.dataset)
    elif args.test_type == 'normal':
        dataset = build_dataset(cfg.val_dataloader.dataset)
    elif args.test_type == 'test_in':
        dataset = build_dataset(cfg.test_in_dataloader.dataset)
    elif args.test_type == 'test_out':
        dataset = build_dataset(cfg.test_out_dataloader.dataset)
    elif args.test_type == 'general':
        dataset = build_dataset(cfg.general_dataloader.dataset)
    else:
        dataset = build_dataset(cfg.val_dataloader.dataset)
    
    if args.test_day is not None:
        
        file_path = dataset.file_path
        label = dataset.label
        file_path_updated = []
        label_updated = []
        for fpth, lbl in zip(file_path, label):
            if 'folder_{}'.format(args.test_day) in fpth:
                file_path_updated.append(fpth)
                label_updated.append(lbl)
        dataset.file_path = file_path_updated
        dataset.label = label_updated
        
 
                
    # extract experiment name from cmd
    config_path = args.config
    exp_cfg_filename = config_path.split('/')[-1].split('.')[0]
    ckp_name = args.ckp_path.replace('/', '_').replace('.pth', '')
    dataset_name = dataset.dataset_name
    log_filename = 'eval_{}_{}_{}_{}_{}.log'.format(timestamp, exp_cfg_filename, args.tag, ckp_name, dataset_name)
    
    # prepare basic text logger
    log_file = osp.join(cfg.work_dir, log_filename)
    log_cfg = dict(log_level='INFO', log_file=log_file)
    log_cfg.setdefault('name', timestamp)
    log_cfg.setdefault('logger_name', 'patchstitcher')
    # `torch.compile` in PyTorch 2.0 could close all user defined handlers
    # unexpectedly. Using file mode 'a' can help prevent abnormal
    # termination of the FileHandler and ensure that the log file could
    # be continuously updated during the lifespan of the runner.
    log_cfg.setdefault('file_mode', 'a')
    logger = MMLogger.get_instance(**log_cfg)
    
    # save some information useful during the training
    runner_info = RunnerInfo()
    runner_info.config = cfg # ideally, cfg should not be changed during process. information should be temp saved in runner_info
    runner_info.logger = logger # easier way: use print_log("infos", logger='current')
    runner_info.rank = rank
    runner_info.distributed = distributed
    runner_info.launcher = cfg.launcher
    runner_info.seed = seed
    runner_info.world_size = world_size
    runner_info.work_dir = cfg.work_dir
    runner_info.timestamp = timestamp
    runner_info.show = args.show
    runner_info.log_filename = log_filename
    runner_info.save_name = args.save_name
    
    if runner_info.show:
        mkdir_or_exist(os.path.join(runner_info.work_dir, runner_info.log_filename + '_show'))
        runner_info.show_dir = os.path.join(runner_info.work_dir, runner_info.log_filename + '_show')
    log_env(cfg, env_cfg, runner_info, logger)
    
    # build model
    model = build_model(cfg.model)
    print_log('Checkpoint Path: {}'.format(cfg.ckp_path), logger='current')
    print_log(model.load_state_dict(torch.load(cfg.ckp_path)['model_state_dict'], strict=False), logger='current')
    model.eval()
    
    if runner_info.distributed:
        torch.cuda.set_device(runner_info.rank)
        model.cuda(runner_info.rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[runner_info.rank], output_device=runner_info.rank,
                                                          find_unused_parameters=cfg.get('find_unused_parameters', False))
        logger.info(model)
    else:
        model.cuda()
    
    
        
    if runner_info.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        val_sampler = None
    
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.val_dataloader.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=val_sampler)

    # build tester
    tester = Tester(
        config=cfg,
        runner_info=runner_info,
        dataloader=val_dataloader,
        model=model)
    
    if args.test_type == 'consistency':
        tester.run_consistency()
    else:
        tester.run_save()

if __name__ == '__main__':
    main()