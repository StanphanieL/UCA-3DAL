import os
import sys
import time
import logging
import glob
import torch


class AverageMeter(object):
    """Track current value, sum, count and running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def create_logger(log_file: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    log_format = '[%(asctime)s  %(levelname)s  %(filename)s  line %(lineno)d  %(process)d]  %(message)s'
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    return logger


def get_logger(cfg):
    """Create a logger whose log file path depends on cfg.task.

    Tasks:
      - train:       <logpath>/train/train-*.log
      - eval / contrastive_eval: <logpath>/result/eval-*.log
      - contrastive: <logpath>/contrastive/contrastive-*.log
      - other:       <logpath>/logs/<task>-*.log
    """

    task = getattr(cfg, 'task', 'train')
    if task == 'train':
        log_file = os.path.join(
            cfg.logpath,
            'train',
            'train-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime())),
        )
    elif task in ('eval', 'contrastive_eval'):
        log_file = os.path.join(
            cfg.logpath,
            'result',
            'eval-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime())),
        )
    elif task == 'contrastive':
        log_file = os.path.join(
            cfg.logpath,
            'contrastive',
            'contrastive-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime())),
        )
    else:
        log_file = os.path.join(
            cfg.logpath,
            'logs',
            '{}-{}.log'.format(task, time.strftime("%Y%m%d_%H%M%S", time.localtime())),
        )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = create_logger(log_file)
    logger.info('************************ Start Logging ************************')
    return logger


def checkpoint_restore(model, optimizer, logpath, epoch: int = 0, dist: bool = False, pretrain_file: str = '', gpu: int = 0):
    """Restore checkpoint.

    Priority:
      1) explicit pretrain_file
      2) <logpath>/latest.pth
      3) max-numbered 00*.pth under logpath
    """

    if not pretrain_file:
        latest_path = os.path.join(logpath, 'latest.pth')
        if os.path.isfile(latest_path):
            pretrain_file = latest_path
        else:
            if epoch > 0:
                pretrain_file = os.path.join(logpath + '%09d' % epoch + '.pth')
                assert os.path.isfile(pretrain_file)
            else:
                cand = sorted(glob.glob(os.path.join(logpath + '00*.pth')))
                if len(cand) > 0:
                    pretrain_file = cand[-1]
                    try:
                        epoch = int(pretrain_file[len(logpath) + 2: -4])
                    except Exception:
                        epoch = 0

    if pretrain_file and len(pretrain_file) > 0 and os.path.isfile(pretrain_file):
        map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
        checkpoint = torch.load(pretrain_file, map_location=map_location)
        model_dict = checkpoint['model']
        optimizer_dict = checkpoint.get('optimizer', None)
        if 'epoch' in checkpoint:
            try:
                epoch = int(checkpoint['epoch'])
            except Exception:
                pass
        else:
            try:
                epoch = int(os.path.splitext(os.path.basename(pretrain_file))[0])
            except Exception:
                pass
        for k, v in model_dict.items():
            if 'module.' in k:
                model_dict = {k[len('module.'):]: v for k, v in model_dict.items()}
            break
        if dist:
            model.module.load_state_dict(model_dict, strict=False)
        else:
            model.load_state_dict(model_dict, strict=False)

        if optimizer is not None and optimizer_dict is not None:
            optimizer.load_state_dict(optimizer_dict)
            for state in optimizer.state.values():
                if state is None:
                    continue
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if dist:
            torch.distributed.barrier()

    return epoch + 1, pretrain_file


def checkpoint_save(model, optimizer, logpath, epoch: int, save_freq: int = 1):
    pretrain_file = os.path.join(logpath + '%09d' % epoch + '.pth')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch,
    }, pretrain_file)
    epoch_prev = epoch - 1
    fd = os.path.join(logpath + '%09d' % epoch_prev + '.pth')
    if os.path.isfile(fd) and epoch_prev % save_freq != 0:
        os.remove(fd)
    return pretrain_file


def checkpoint_save_newest(model, optimizer, logpath, epoch: int, save_freq: int = 1):
    """Save numbered checkpoint and update latest.pth.

    - always update <logpath>/latest.pth with current epoch
    - every ``save_freq`` epochs, also save a numbered snapshot
    """

    latest_path = os.path.join(logpath, 'latest.pth')
    os.makedirs(logpath, exist_ok=True)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch,
    }
    torch.save(state, latest_path)

    if save_freq is not None and save_freq > 0 and epoch % save_freq == 0:
        numbered_path = os.path.join(logpath + '%09d' % epoch + '.pth')
        torch.save(state, numbered_path)

    return latest_path


def print_error(message, user_fault: bool = False):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    if user_fault:
        sys.exit(2)
    sys.exit(-1)