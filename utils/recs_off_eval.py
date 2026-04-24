from .ris_off_eval import ris_off_eval, metric_table
from .vg_off_eval import vg_off_eval


def recs_off_eval(pred_ris_path, pred_vg_path, gt_ris_path, gt_vg_path, gt_suffix='.png', verbose=True):
    ris_metrics = ris_off_eval(pred_ris_path, gt_ris_path, gt_suffix, False)
    vg_metrics = vg_off_eval(pred_vg_path, gt_vg_path, False)
    metrics = ris_metrics + vg_metrics
    metrics.append(ris_metrics[-1] + vg_metrics[-1])

    if verbose:
        print(metric_table(
            metrics, ['RIS_P@.5', 'RIS_P@.6', 'RIS_P@.7', 'RIS_P@.8', 'RIS_P@.9', 'RIS_oIoU', 'RIS_mIoU', 'RIS_SUM',
                      'VG_P@.5', 'VG_P@.6', 'VG_P@.7', 'VG_P@.8', 'VG_P@.9', 'VG_oIoU', 'VG_mIoU', 'VG_SUM',
                      'RECS_SUM']))
    else:
        return metrics


def main():
    pred_ris_path = input('please enter your ris prediction dir: ')
    pred_vg_path = input('please enter your vg prediction dir: ')
    label_path = input('please enter your groundtruth selection from 1: RefDIOR-Val 2: RefDIOR-Test\nYour choice: ')

    if '1' == label_path:
        gt_ris_path = '/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/masks'
        gt_vg_path = '/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/phrase_txts/val.json'
        gt_suffix = '.png'
    elif '2' == label_path:
        gt_ris_path = '/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/masks'
        gt_vg_path = '/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/phrase_txts/test.json'
        gt_suffix = '.png'
    else:
        raise NotImplementedError

    recs_off_eval(pred_ris_path, pred_vg_path, gt_ris_path, gt_vg_path, gt_suffix)


if __name__ == '__main__':
    main()