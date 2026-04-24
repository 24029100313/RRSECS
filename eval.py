import argparse, torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.utils import metric_table, make_save_path
from utils.dist_helper import setup_distributed
from rsfm.dataset import create_test_dataset, datasets_info
from rsfm.builder import test_model_builder
from rsfm.engine import eval_engine_builder, predict_engine_builder


cudnn.enabled = True
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--task', type=str, default='eval', choices=['eval', 'predict'])
parser.add_argument('--weight-path', type=str, default=None)
parser.add_argument('--output-path', type=str, default=None)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--tta', action='store_true')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    # initialize
    args = parser.parse_args()
    rank, world_size = setup_distributed(port=args.port)

    # create and load model from weight path
    model, vision_task, num_classes, dataset_name, img_size = test_model_builder(args)

    # create test dataset
    testset = create_test_dataset(args.task, dataset_name, 'val', img_size)
    testsampler = torch.utils.data.distributed.DistributedSampler(testset)
    testloader = DataLoader(testset, batch_size=128 if vision_task == 'scene classification' else 1,
                            pin_memory=False, num_workers=1, drop_last=False, sampler=testsampler, shuffle=False)

    # evaluation or prediction
    if args.task == 'eval':
        metrics, metrics_name = eval_engine_builder(vision_task, model, testloader, num_classes, tta=args.tta)
        need_perclass, classes_name = False, None
        if vision_task in ['semantic segmentation', 'change detection']:
            need_perclass, classes_name = True, datasets_info[dataset_name]['classes_name']
        if rank == 0:
            if args.tta:
                print('Using multi-scale with horizontal flipping TTA')
            eval_results = metric_table(metrics, metrics_name, need_perclass, classes_name)
            print('Evaluation Results: \n{}'.format(eval_results))

    elif args.task == 'predict':
        make_save_path(args, rank, vision_task)
        predict_engine_builder(vision_task, model, testloader, args)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
