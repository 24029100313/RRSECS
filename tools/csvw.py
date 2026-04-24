import os, csv


def get_best(path):
    for file in os.listdir(path):
        if 'log' in file:
            alls = []
            infos = open(path + '/' + file, 'r').readlines()
            for info in infos:
                need = info.strip()
                if '|' in need and 'Evaluating' not in need and 'oIoU' not in need:
                    all = need.split('|')
                    fff = [float(i) for i in all[1:8]]
                    alls.append(fff)
            alls = sorted(alls, key=lambda x:x[-2])
            return alls[-1]


paths = [
    'exps/vg/RefCOCO_VG/convnext_tiny_transvg_640',
    'exps/vg/RefCOCO_VG/convnext_tiny_vltvg_640',
    'exps/vg/RefCOCO_VG/convnext_tiny_pseudoq_640',
    'exps/vg/RefCOCO_VG/convnext_tiny_qrnet_640',
    'exps/vg/RefCOCO_VG/convnext_tiny_dmdetr_640',
    'exps/vg/RefCOCO_VG/convnext_tiny_transcp_640',
    'exps/vg/RefCOCO_VG/convnext_tiny_lqvg_640',
    'exps/vg/RefCOCO_VG/convnext_tiny_lpva_640',
]

a = []
for path in paths:
    a.append(get_best(path))

with open('exps/vg/RefCOCO_VG/convnext640_output.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(a)


# a = []
# for path in paths:
#     for file in os.listdir(path):
#         if 'best' in file:
#             a.append(path + '/' + file)
# print(a)