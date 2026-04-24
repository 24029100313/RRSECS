import torch
from tqdm import tqdm

def ms_model(paths, save_path):
    '''
    Args:
        paths: list to save weight path
        save_path: model soup merge path

    Returns:
    '''

    weights = []
    for path in paths:
        print('load state_dict of: {}'.format(path))
        weights.append(torch.load(path)['model'])

    for k in tqdm(weights[0].keys()):
        v = weights[0][k]
        for weight in weights[1:]:
            v += weight[k]
        if torch.is_floating_point(v):
            v /= len(paths)
        else:
            gd = v / len(paths)
            v = gd.long()

    torch.save({'model': weights[0]}, save_path)


model_name = '/path/to/your/local/dir'

proposals = [
    model_name + '/epoch_18.pth',
    model_name + '/epoch_19.pth',
    model_name + '/epoch_20.pth',
    model_name + '/epoch_21.pth',
    model_name + '/epoch_22.pth',
    model_name + '/epoch_23.pth',
    model_name + '/epoch_24.pth'
]



ms_model(proposals, model_name + '/ep18toep24_ms7.pth')



