import argparse
import onnx
import onnxsim
import torch
import warnings
from pathlib import Path

from summary import (file_size, attempt_load, Detect, End2End)


def export_onnx(model, im, file, opset, simplify, end2end):
    # YOLOv5 ONNX export
    try:

        print(f'\nstarting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')

        torch.onnx.export(
            model,
            im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
            if end2end else ['output'],
            dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        onnx.save(model_onnx, f)

        # Simplify
        if simplify and not end2end:
            try:
                print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=False,
                                                     input_shapes=None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'simplifier failure: {e}')
        print(f'export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        print(f'export failure: {e}')


@torch.no_grad()
def run(
        weights='yolov7.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        end2end=False,  # TRT: add EfficientNMS_TRT to model
        with_preprocess=False,  # TRT: add preprocess to model (BGR2RGB and divide by 255)
        topk_all=100,  # TRT: topk for every image to keep
        iou_thres=0.45,  # TRT: IoU threshold
        conf_thres=0.25,  # TRT: confidence threshold
):
    file = Path(weights)
    assert file.exists(), f'{file} is not exists'

    # Load PyTorch model
    device = torch.device(device)
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model
    nc, names = model.nc, model.names  # number of classes, class names

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    assert nc == len(names), f'Model class count {nc} != len(names) {len(names)}'

    # Input
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False
            m.onnx_dynamic = False
            m.export = True
    if end2end:
        model = End2End(model, topk_all, iou_thres, conf_thres, device, with_preprocess=with_preprocess)
    y, simplify = None, False if end2end else simplify
    for _ in range(2):
        y = model(im)  # dry runs
    shape = tuple(y[0].shape)  # model output shape
    print(f"\nPyTorch: starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports

    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    fstr = export_onnx(model, im, file, opset, simplify, end2end)

    return fstr


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--end2end', action='store_true', help='TRT: add EfficientNMS_TRT to model')
    parser.add_argument('--with-preprocess', action='store_true',
                        help='TRT: add preprocess to model (BGR2RGB and divide by 255)')
    parser.add_argument('--topk-all', type=int, default=100, help='TRT: topk for every image to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='TRT: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='TRT: confidence threshold')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
