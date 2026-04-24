#!/bin/bash

python -m torch.distributed.run \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    eval.py \
    --port $2 \
    ${@:3}



#wps=(
#'exps/vg/RefDIOR_VG/sa/swin/swin_tiny_transvg/RefDIOR_VG.swin_tiny.None.TransVGHead.4xb8.img512.ep50.preimagenet.best68.63.pth'
#'exps/vg/RefDIOR_VG/sa/swin/swin_tiny_vltvg/RefDIOR_VG.swin_tiny.None.VLTVGHead.4xb8.img512.ep50.preimagenet.best70.71.pth'
#'exps/vg/RefDIOR_VG/sa/swin/swin_tiny_pseudoq/RefDIOR_VG.swin_tiny.None.PseudoQHead.4xb8.img512.ep50.preimagenet.best70.37.pth'
#'exps/vg/RefDIOR_VG/sa/swin/swin_tiny_qrnet/RefDIOR_VG.swin_tiny.QMF.QRNetHead.4xb8.img512.ep50.preimagenet.best70.83.pth'
#'exps/vg/RefDIOR_VG/sa/swin/swin_tiny_dmdetr/RefDIOR_VG.swin_tiny.None.DynamicMDETRHead.4xb8.img512.ep50.preimagenet.best70.36.pth'
#'exps/vg/RefDIOR_VG/sa/swin/swin_tiny_transcp/RefDIOR_VG.swin_tiny.None.TransCPHead.4xb8.img512.ep50.preimagenet.best55.4.pth'
#'exps/vg/RefDIOR_VG/sa/swin/swin_tiny_lqvg/RefDIOR_VG.swin_tiny.None.LQVGHead.4xb8.img512.ep50.preimagenet.best83.39.pth'
#'exps/vg/RefDIOR_VG/sa/swin/swin_tiny_lpva/RefDIOR_VG.swin_tiny.None.LPVAHead.4xb8.img512.ep50.preimagenet.best72.13.pth'
#'exps/vg/RefDIOR_VG/sa/convnext/convnext_tiny_transvg/RefDIOR_VG.convnext_tiny.None.TransVGHead.4xb8.img512.ep50.preimagenet.best68.23.pth'
#'exps/vg/RefDIOR_VG/sa/convnext/convnext_tiny_vltvg/RefDIOR_VG.convnext_tiny.None.VLTVGHead.4xb8.img512.ep50.preimagenet.best69.85.pth'
#'exps/vg/RefDIOR_VG/sa/convnext/convnext_tiny_pseudoq/RefDIOR_VG.convnext_tiny.None.PseudoQHead.4xb8.img512.ep50.preimagenet.best70.27.pth'
#'exps/vg/RefDIOR_VG/sa/convnext/convnext_tiny_qrnet/RefDIOR_VG.convnext_tiny.QMF.QRNetHead.4xb8.img512.ep50.preimagenet.best68.48.pth'
#'exps/vg/RefDIOR_VG/sa/convnext/convnext_tiny_dmdetr/RefDIOR_VG.convnext_tiny.None.DynamicMDETRHead.4xb8.img512.ep50.preimagenet.best72.1.pth'
#'exps/vg/RefDIOR_VG/sa/convnext/convnext_tiny_transcp/RefDIOR_VG.convnext_tiny.None.TransCPHead.4xb8.img512.ep50.preimagenet.best54.74.pth'
#'exps/vg/RefDIOR_VG/sa/convnext/convnext_tiny_lqvg/RefDIOR_VG.convnext_tiny.None.LQVGHead.4xb8.img512.ep50.preimagenet.best82.88.pth'
#'exps/vg/RefDIOR_VG/sa/convnext/convnext_tiny_lpva/RefDIOR_VG.convnext_tiny.None.LPVAHead.4xb8.img512.ep50.preimagenet.best73.34.pth'
#'exps/vg/RefDIOR_VG/sa/vmamba/vmamba_tiny_transvg/RefDIOR_VG.vmamba_tiny.None.TransVGHead.4xb8.img512.ep50.preimagenet.best71.41.pth'
#'exps/vg/RefDIOR_VG/sa/vmamba/vmamba_tiny_vltvg/RefDIOR_VG.vmamba_tiny.None.VLTVGHead.4xb8.img512.ep50.preimagenet.best74.97.pth'
#'exps/vg/RefDIOR_VG/sa/vmamba/vmamba_tiny_pseudoq/RefDIOR_VG.vmamba_tiny.None.PseudoQHead.4xb8.img512.ep50.preimagenet.best74.36.pth'
#'exps/vg/RefDIOR_VG/sa/vmamba/vmamba_tiny_qrnet/RefDIOR_VG.vmamba_tiny.QMF.QRNetHead.4xb8.img512.ep50.preimagenet.best75.94.pth'
#'exps/vg/RefDIOR_VG/sa/vmamba/vmamba_tiny_dmdetr/RefDIOR_VG.vmamba_tiny.None.DynamicMDETRHead.4xb8.img512.ep50.preimagenet.best73.71.pth'
#'exps/vg/RefDIOR_VG/sa/vmamba/vmamba_tiny_transcp/RefDIOR_VG.vmamba_tiny.None.TransCPHead.4xb8.img512.ep50.preimagenet.best54.57.pth'
#'exps/vg/RefDIOR_VG/sa/vmamba/vmamba_tiny_lqvg/RefDIOR_VG.vmamba_tiny.None.LQVGHead.4xb8.img512.ep50.preimagenet.best83.32.pth'
#'exps/vg/RefDIOR_VG/sa/vmamba/vmamba_tiny_lpva/RefDIOR_VG.vmamba_tiny.None.LPVAHead.4xb8.img512.ep50.preimagenet.best75.52.pth'
#)


#wps=(
#'exps/vg/RefDIOR_VG/wosa/swin/swin_tiny_transvg_wosa/RefDIOR_VG.swin_tiny.None.TransVGHead.4xb8.img512.ep50.preimagenet.best66.81.pth'
#'exps/vg/RefDIOR_VG/wosa/swin/swin_tiny_vltvg_wosa/RefDIOR_VG.swin_tiny.None.VLTVGHead.4xb8.img512.ep50.preimagenet.best70.21.pth'
#'exps/vg/RefDIOR_VG/wosa/swin/swin_tiny_pseudoq_wosa/RefDIOR_VG.swin_tiny.None.PseudoQHead.4xb8.img512.ep50.preimagenet.best68.98.pth'
#'exps/vg/RefDIOR_VG/wosa/swin/swin_tiny_qrnet_wosa/RefDIOR_VG.swin_tiny.QMF.QRNetHead.4xb8.img512.ep50.preimagenet.best69.1.pth'
#'exps/vg/RefDIOR_VG/wosa/swin/swin_tiny_dmdetr_wosa/RefDIOR_VG.swin_tiny.None.DynamicMDETRHead.4xb8.img512.ep50.preimagenet.best69.42.pth'
#'exps/vg/RefDIOR_VG/wosa/swin/swin_tiny_transcp_wosa/RefDIOR_VG.swin_tiny.None.TransCPHead.4xb8.img512.ep50.preimagenet.best59.29.pth'
#'exps/vg/RefDIOR_VG/wosa/swin/swin_tiny_lqvg_wosa/RefDIOR_VG.swin_tiny.None.LQVGHead.4xb8.img512.ep50.preimagenet.best81.86.pth'
#'exps/vg/RefDIOR_VG/wosa/swin/swin_tiny_lpva_wosa/RefDIOR_VG.swin_tiny.None.LPVAHead.4xb8.img512.ep50.preimagenet.best71.28.pth'
#'exps/vg/RefDIOR_VG/wosa/convnext/convnext_tiny_transvg_wosa/RefDIOR_VG.convnext_tiny.None.TransVGHead.4xb8.img512.ep50.preimagenet.best67.58.pth'
#'exps/vg/RefDIOR_VG/wosa/convnext/convnext_tiny_vltvg_wosa/RefDIOR_VG.convnext_tiny.None.VLTVGHead.4xb8.img512.ep50.preimagenet.best68.67.pth'
#'exps/vg/RefDIOR_VG/wosa/convnext/convnext_tiny_pseudoq_wosa/RefDIOR_VG.convnext_tiny.None.PseudoQHead.4xb8.img512.ep50.preimagenet.best69.53.pth'
#'exps/vg/RefDIOR_VG/wosa/convnext/convnext_tiny_qrnet_wosa/RefDIOR_VG.convnext_tiny.QMF.QRNetHead.4xb8.img512.ep50.preimagenet.best67.18.pth'
#'exps/vg/RefDIOR_VG/wosa/convnext/convnext_tiny_dmdetr_wosa/RefDIOR_VG.convnext_tiny.None.DynamicMDETRHead.4xb8.img512.ep50.preimagenet.best65.95.pth'
#'exps/vg/RefDIOR_VG/wosa/convnext/convnext_tiny_transcp_wosa/RefDIOR_VG.convnext_tiny.None.TransCPHead.4xb8.img512.ep50.preimagenet.best64.18.pth'
#'exps/vg/RefDIOR_VG/wosa/convnext/convnext_tiny_lqvg_wosa/RefDIOR_VG.convnext_tiny.None.LQVGHead.4xb8.img512.ep50.preimagenet.best82.18.pth'
#'exps/vg/RefDIOR_VG/wosa/convnext/convnext_tiny_lpva_wosa/RefDIOR_VG.convnext_tiny.None.LPVAHead.4xb8.img512.ep50.preimagenet.best71.79.pth'
#'exps/vg/RefDIOR_VG/wosa/vmamba/vmamba_tiny_transvg_wosa/RefDIOR_VG.vmamba_tiny.None.TransVGHead.4xb8.img512.ep50.preimagenet.best67.66.pth'
#'exps/vg/RefDIOR_VG/wosa/vmamba/vmamba_tiny_vltvg_wosa/RefDIOR_VG.vmamba_tiny.None.VLTVGHead.4xb8.img512.ep50.preimagenet.best72.75.pth'
#'exps/vg/RefDIOR_VG/wosa/vmamba/vmamba_tiny_pseudoq_wosa/RefDIOR_VG.vmamba_tiny.None.PseudoQHead.4xb8.img512.ep50.preimagenet.best72.15.pth'
#'exps/vg/RefDIOR_VG/wosa/vmamba/vmamba_tiny_qrnet_wosa/RefDIOR_VG.vmamba_tiny.QMF.QRNetHead.4xb8.img512.ep50.preimagenet.best75.46.pth'
#'exps/vg/RefDIOR_VG/wosa/vmamba/vmamba_tiny_dmdetr_wosa/RefDIOR_VG.vmamba_tiny.None.DynamicMDETRHead.4xb8.img512.ep50.preimagenet.best71.13.pth'
#'exps/vg/RefDIOR_VG/wosa/vmamba/vmamba_tiny_transcp_wosa/RefDIOR_VG.vmamba_tiny.None.TransCPHead.4xb8.img512.ep50.preimagenet.best69.79.pth'
#'exps/vg/RefDIOR_VG/wosa/vmamba/vmamba_tiny_lqvg_wosa/RefDIOR_VG.vmamba_tiny.None.LQVGHead.4xb8.img512.ep50.preimagenet.best82.18.pth'
#'exps/vg/RefDIOR_VG/wosa/vmamba/vmamba_tiny_lpva_wosa/RefDIOR_VG.vmamba_tiny.None.LPVAHead.4xb8.img512.ep50.preimagenet.best74.14.pth'
#)


#for wp in "${wps[@]}"
#do
#  python -m torch.distributed.run \
#    --nproc_per_node=$1 \
#    --master_addr=localhost \
#    --master_port=$2 \
#    eval.py \
#    --weight-path=$wp \
#    --port $2 \
#    ${@:3}
#done