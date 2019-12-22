import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy, data_prefetcher


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    prefetcher = data_prefetcher(data_loader)
    end_time = time.time()
    inputs, targets = prefetcher.next()
    i = 0
    while inputs is not None:
        data_time.update(time.time() - end_time)

        # if not opt.no_cuda:
        #     targets = targets.cuda(non_blocking=True)
        #     inputs = inputs.cuda(non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[-1]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        inputs, targets = prefetcher.next()
        i += 1
    
    # epoch message
    time_message = [divmod(int(batch_time.sum), 60), divmod(int(data_time.sum), 60)]
    print('-------------------------------------------------------')
    print('Train Epoch: [{0}]\t'
          'Time {tmessage[0][0]}:{tmessage[0][1]}\t'
          'Data {tmessage[1][0]}:{tmessage[1][1]}\t'
          'Loss {loss:.4f}\t'
          'Acc {acc:.3f}%\t'
          'LR {lr}'.format(
            epoch,
            tmessage=time_message,
            loss=losses.avg,
            acc=accuracies.avg * 100,
            lr=optimizer.param_groups[-1]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[-1]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
