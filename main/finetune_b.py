import argparse
import torch
import torch.backends.cudnn as cudnn
from config import cfg
import os.path as osp

# ddp
import torch.distributed as dist
from common.utils.distribute_utils import (
    init_distributed_mode, is_main_process, set_seed
)
import torch.distributed as dist
from mmcv.runner import get_dist_info

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--master_port', type=int, dest='master_port')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--config', type=str, default='./config/config_base.py')
    parser.add_argument('--ft_lr_factor', type=float, default=0.1, description='fine-tune learning rate factor')
    parser.add_argument('--pre_trained_model_path', type=str, default='../pretrained_models/smpler_x_h32.pth.tar')
    parser.add_argument('--freeze_vit_backbone', type=bool, default=False, description='freeze fine-tuning ViT backbone')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    config_path = osp.join('./config', args.config)
    cfg.get_config_fromfile(config_path)
    cfg.update_config(args.num_gpus, args.exp_name)
    pre_trained_model_path = args.pre_trained_model_path
    fine_tune_lr_factor = args.ft_lr_factor
    freeze_vit_backbone = args.freeze_vit_backbone

    cudnn.benchmark = True
    set_seed(2023)

    # ddp by default in this branch
    distributed, gpu_idx = init_distributed_mode(args.master_port)
    from base import Trainer
    trainer = Trainer(distributed, gpu_idx)
    
    # ddp
    if distributed:
        trainer.logger_info('### Set DDP ###')
        trainer.logger.info(f'Distributed: {distributed}, init done {gpu_idx}')
    else:
        raise Exception("DDP not setup properly")
    
    # Load pre-trained model
    pre_trained_model_path = "path_to_pretrained_model.pth"
    if osp.exists(pre_trained_model_path):
        checkpoint = torch.load(pre_trained_model_path)
        trainer.model.load_state_dict(checkpoint['network'])
        trainer.logger_info(f"Loaded pre-trained model from {pre_trained_model_path}")
    
    # Freeze the ViT backbone layers
    if freeze_vit_backbone:
        trainer.logger_info("Freezing ViT backbone layers")
        for param in trainer.model.backbone.parameters():
            param.requires_grad = False

    # Adjust learning rate for fine-tuning
    print("train_lr before fine-tuning: ", cfg.train_lr)
    cfg.train_lr *= fine_tune_lr_factor
    print("train_lr after fine-tuning: ", cfg.train_lr)

    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=cfg.train_lr)


    trainer.logger_info(f"Using {cfg.num_gpus} GPUs, batch size {cfg.train_batch_size} per GPU.")
    
    trainer._make_batch_generator()
    trainer._make_model()

    trainer.logger_info('### Set some hyper parameters ###')
    for k in cfg.__dict__:
        trainer.logger_info(f'set {k} to {cfg.__dict__[k]}')
        trainer.logger_info(f'train with train_3d={cfg.trainset_3d}')
        trainer.logger_info(f'train with train_2d={cfg.trainset_2d}')
        trainer.logger_info(f'train with trainset_humandata={cfg.trainset_humandata}')

    trainer.logger_info('### Start fine-tuning ###')

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        # ddp, align random seed between devices
        trainer.batch_generator.sampler.set_epoch(epoch)

        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss= trainer.model(inputs, targets, meta_info, 'train')
            loss_mean = {k: loss[k].mean() for k in loss}
            loss_sum = sum(loss_mean[k] for k in loss_mean)
            
            # backward
            loss_sum.backward()
            trainer.optimizer.step()
            trainer.scheduler.step()
            
            trainer.gpu_timer.toc()
            if (itr + 1) % cfg.print_iters == 0:
                # loss of all ranks
                rank, world_size = get_dist_info()
                loss_print = loss_mean.copy()
                for k in loss_print:
                    dist.all_reduce(loss_print[k]) 
                
                total_loss = 0
                for k in loss_print:
                    loss_print[k] = loss_print[k] / world_size
                    total_loss += loss_print[k]
                loss_print['total'] = total_loss
                    
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time,
                        trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss_print.items()]
                trainer.logger_info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        # save model ddp, save model.module on rank 0 only
        save_epoch = getattr(cfg, 'save_epoch', 10)
        if is_main_process() and (epoch % save_epoch == 0 or epoch == cfg.end_epoch - 1):
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)

        dist.barrier()

if __name__ == "__main__":
    main()