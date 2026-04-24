import argparse, torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.utils import metric_table, make_save_path, count_params
from utils.dist_helper import setup_distributed
from rsfm.dataset import create_test_dataset, datasets_info
from rsfm.classification import swin_tiny, swin_small
from rsfm.engine import eval_engine_builder, predict_engine_builder
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


cudnn.enabled = True
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--task', type=str, default='eval', choices=['eval', 'predict'])
parser.add_argument('--weight-path', type=str, required=True)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--tta', action='store_true')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    # initialize
    args = parser.parse_args()
    rank, world_size = setup_distributed(port=args.port)

    # create and load model from weight path
    dataset_name = 'ImageNet1k'
    img_size = 224
    num_classes = 1000
    vision_task = 'scene classification'
    model = swin_tiny(img_size=img_size, num_classes=num_classes)
    if rank == 0:
        print('Loading from {}\n'.format(args.weight_path))
        print('Total params: {:.2f}M\n'.format(count_params(model)))

    model.cuda()
    model_without_ddp = model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)

    checkpoint = torch.load(args.weight_path, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    # # create test dataset
    # transform = transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    #                                 transforms.CenterCrop(224),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # root = '/root/lxq/RSFM_CLS_Datasets/Natural/ImageNet1k/val'
    # testset = datasets.ImageFolder(root, transform=transform)

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
        make_save_path(args, rank)
        predict_engine_builder(vision_task, model, testloader, args)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
