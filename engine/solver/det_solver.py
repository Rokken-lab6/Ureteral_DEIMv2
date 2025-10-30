"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches, 
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        n_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_parameters}')

        top1 = 0.0
        best_stat = {'epoch': -1, 'best_f1': 0.0, 'best_f1_threshold': 0.0}
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, _ = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.cfg.distance_threshold,
            )
            best_f1 = float(test_stats.get('best_f1', 0.0))
            best_threshold = float(test_stats.get('best_f1_threshold', 0.0))
            best_stat['epoch'] = self.last_epoch
            best_stat['best_f1'] = best_f1
            best_stat['best_f1_threshold'] = best_threshold
            top1 = best_f1
            print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, _ = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.cfg.distance_threshold,
            )

            best_f1 = float(test_stats.get('best_f1', 0.0))
            best_threshold = float(test_stats.get('best_f1_threshold', 0.0))
            f1_per_threshold = test_stats.get('f1_per_threshold', {})

            if self.writer and dist_utils.is_main_process():
                self.writer.add_scalar('Test/best_f1', best_f1, epoch)
                self.writer.add_scalar('Test/best_f1_threshold', best_threshold, epoch)
                for thr, value in f1_per_threshold.items():
                    self.writer.add_scalar(f'Test/f1@{thr:.2f}', value, epoch)

            previous_best = best_stat.get('best_f1', float('-inf'))
            if best_f1 > previous_best:
                best_stat['epoch'] = epoch
                best_stat['best_f1'] = best_f1
                best_stat['best_f1_threshold'] = best_threshold
            else:
                best_stat['best_f1'] = max(previous_best, best_f1)

            if best_stat.get('best_f1', 0.0) > top1:
                top1 = best_stat['best_f1']
                best_stat_print['epoch'] = best_stat['epoch']
                best_stat_print['best_f1_threshold'] = best_stat.get('best_f1_threshold', 0.0)
                if self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

            best_stat_print['best_f1'] = best_stat.get('best_f1', 0.0)
            best_stat_print['best_f1_threshold'] = best_stat.get('best_f1_threshold', 0.0)
            print(f'best_stat: {best_stat_print}')  # global best

            if best_stat.get('epoch', -1) == epoch and self.output_dir:
                if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if best_f1 > top1:
                        top1 = best_f1
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                else:
                    top1 = max(best_f1, top1)
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

            elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                best_stat = {'epoch': -1, 'best_f1': 0.0, 'best_f1_threshold': 0.0}
                self.ema.decay -= 0.0001
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'test_best_f1': best_f1,
                'test_best_f1_threshold': best_threshold,
                'test_best_f1_conf_threshold': best_threshold,
                **{f'test_f1@{thr:.2f}': value for thr, value in f1_per_threshold.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, _ = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            self.cfg.distance_threshold,
        )

        print(f"Validation best F1: {test_stats.get('best_f1', 0.0):.4f} @ threshold {test_stats.get('best_f1_threshold', 0.0):.2f}")

        if self.output_dir and dist_utils.is_main_process():
            log_path = self.output_dir / "val_metrics.json"
            with log_path.open("w") as f:
                json.dump(test_stats, f)

        return
