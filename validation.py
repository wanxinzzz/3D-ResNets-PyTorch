import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy, data_prefetcher


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    prefetcher = data_prefetcher(data_loader)
    end_time = time.time()
    inputs, targets = prefetcher.next()
    i = 0
    with torch.no_grad():
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

            batch_time.update(time.time() - end_time)
            end_time = time.time()

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
    print('Val Epoch: [{0}]\t'
          'Time {tmessage[0][0]}:{tmessage[0][1]}\t'
          'Data {tmessage[1][0]}:{tmessage[1][1]}\t'
          'Loss {loss:.4f}\t'
          'Acc {acc:.3f}%'.format(
            epoch,
            tmessage=time_message,
            loss=losses.avg,
            acc=accuracies.avg * 100))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg
