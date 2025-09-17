# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
#导入python包
import argparse #1、导入python安装库eg.python,os等
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()#2、定义了一些路径  file当前执行的detect.py的路径，得到其绝对路径
ROOT = FILE.parents[0]  # YOLOv5 root directory #parents[0]获得detect.py的父目录，即整个yolov5项目的路径
if str(ROOT) not in sys.path: #判断yolov5路径是否存在模块的查询路径   sys.path:模块的查询路径列表 sys.path路径下存在yolov5文件夹路径
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 绝对路径转换成相对路径

from models.common import DetectMultiBackend #3、导入相对路径下的一些模块eg.DetectMultiBackend这个类
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode() #run函数分成6个部分
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):#第1部分：对source额外传入的东西进行判断，就是对Terminal手动输入命令进行判断
    source = str(source) #source就是在命令行传入的参数 eg.data\\images\\bus.jpg代表这个路径，str代表强制把source路径转为字符串类型
    save_img = not nosave and not source.endswith('.txt')  # save inference images 需要保存预测结果，true+true e_img这个标志位预测结果需要保存下来
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) #Ctrl+IMG_FORMATS，直接跳到dataloaders.py。#判断传入路径是否是文件地址，suffix表示后缀的意思，[1:]从j开始所有，即jpg，判断jpg是否在IMG_FORMATS + VID_FORMATS这两个变量中
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))#判断我们给出的地址data\\images\\bus.jpg是否为网络流地址，或是网络图片地址
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)#isnumeric判断source路径是否为数字。数字0为打开电脑上的第一个摄像头。判断传入地址是否为摄像头，.streams结尾，网络流地址？
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download #如果source的路径是网络流地址，根据网络流地址下载图片或视频

    # Directories #第2部分：新建保存结果的文件夹
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run #project+name参数拼接起来，increment_path增量路径
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir #在save_dir下新建一个labels的文件夹，参数save_txt默认为false

    # Load model #第3部分：加载模型的权重
    device = select_device(device) #根据环境，选择加载模型的设备，CPU OR GPU
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) #创建了DetectMultiBackend这个类，传入些参数。选择模型的后端框架，Ctrl+DetectMultiBackend,直接跳转到common.py.深度学习框架的选择
    stride, names, pt = model.stride, model.names, model.pt #得到了些属性，步长，名字，pt。加载模型读取一些值，模型的步长，模型能检测出来的类别名，模型是否为pytorch的模型类型
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader #第4部分：定义了一个Dataloader模块，负责加载待预测的图片
    bs = 1  # batch_size
    if webcam: #判断webcam是否为true，判定为false
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:  #false
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else: # true 加载图片文件
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)#初始化LoadImages这个对象，专门负责加载图像的模块
    vid_path, vid_writer = [None] * bs, [None] * bs #bs:batch_size

    # Run inference #第5部分：执行模型推理的过程，把图片输入模型，产生预测结果，把最后的检测框画出来
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    # warmup热身，初始化了一张空白的图片，传入到模型中，执行了一次前馈传播，相当于随便给gpu一张图片让gpu先跑一下
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile()) #seen,windows，dt预测之前提前定义好的变量，用来存储中间结果信息
    for path, im, im0s, vid_cap, s in dataset: #path, im, im0s, vid_cap, s in dataset把我们自己的图片依次传给模型，让模型依次去预测
     #遍历for循环的时候，得到图片的路径 "F:\\yolov5-master\\data\\images\\bus.jpg" im:resize后的图片[3,640,480],im0s；原图[1080,810]
     #vid_cap：空 s：打印信息
        with dt[0]: #预处理
            im = torch.from_numpy(im).to(model.device) #torch.Size([3,640,480])
            #dataloader中得到的图片是numpy格式的数组，如果想要输入到模型中运算的话必须转成pytorch支持的tensor类型，在把它放到cpu上（to(model.device)）
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 #判断模型是否用到半精度，没有，即为float类型
            im /= 255  # 0 - 255 to 0.0 - 1.0 #把输入图像归一化
            if len(im.shape) == 3: #判断输入图像的尺寸是否为3
                im = im[None]  # expand for batch dim  #扩增一下batch维度，张量变成了torch.Size([1,3,640,480])


        # Inference #预测
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            #visualize参数，调用run函数的时候传进来这个参数，默认是false，如果是true的话，会把推断过程中的特征图也保存下来
            pred = model(im, augment=augment, visualize=visualize) #torch.Size([1,18900,85])
            #augment也是传进来的一个参数，表示模型推断的过程中是否做数据增强，增强可能会对推断结果有帮助，但是会降低模型额运行速度
            #pred为模型预测结果，模型预测检测框是18900，后面会进一步过滤，85=4个坐标信息+1个置信度信息+80个类别的概率值

        # NMS #非极大值抑制
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det = max_det ) #pred包含了一个batch中的所有图片
         #1，5，6 [6.72000e+02, 3.95000e+02, 8.1000e+02, 8.78000e+02, 8.96172e-01, 0.0000e+00]
        # 从18900个检测框经过滤，最后剩下5个检测框（目标），6的前四个值表示的坐标信息（672，395，810，878）xyxy，置信度信息0.89，类别信息：0（人）

        #Second-stage classifier (optional)
        #pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions #把所有的检测框画到原图中
        for i, det in enumerate(pred):  # per image #遍历一个batch中的每个图片，这里的det表示5个检测框的预测信息，torch_Size([5,6])
            seen += 1 #seen技术的功能，每处理一张图片+1
            if webcam:  # batch_size >= 1 #判断是否为webcam，false
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                # 执行else，赋了几个值，p这次循环中所使用的路径，frame:判断dataset中是否具有frame属性，如果没有就是0

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg #图片的保存路径，存贮路径+图片名称拼接起来，"runs\\detect\\exp3","bus.jpg"
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt txt的存贮路径，默认是不存txt的
            s += '%gx%g ' % im.shape[2:]  # print string 把s拼接了一些信息，%g*%g表示图片的尺寸，打印信息里的640*480
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh #获得原图的宽和高的大小
            imc = im0.copy() if save_crop else im0  # for save_crop #判断是否需要把检测框的区域裁剪下来，单独保存成一张图片
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)) #定义了专门绘图的工具。Annotator是个类，plots.py
            #把参数传进去，line_thickness：画线检测框的线条粗细，默认是3个像素点的粗细， str(name):所预测的标签的标签名，80个标签的类别
            if len(det): #det这个变量，有5个框，每个框有6个值。判断是否有框
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() #坐标映射（640，480）（1080，810）
                #预测出来的值是基于640*480，不能画到原图1080*810上，坐标映射，方便在原图上画框

                # Print results
                for c in det[:, 5].unique(): #遍历det这个变量，det这个变量存储了5个框
                    n = (det[:, 5] == c).sum()  # detections per class #统计所有框的类别
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string #把统计所有框的类别添加到s这个变量上
                    #打印信息4 person， 1 bus

                # Write results #保存结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file #把结果保存为txt格式，默认为false
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image #把结果画到图片上保存
                        c = int(cls)  # integer class #获得类别
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #hide_labels隐藏标签，置信度
                        color_dict = {'0': [0, 0, 255], '1': [0, 255, 255], '2': [0, 255, 0]}  # Rgb3个值：B G R
                        if names[int(cls)] == 'target':  # 根据训练的文件.yaml中的name类中的名称修改
                            color_single = color_dict['0']
                        elif names[int(cls)] == 'island':
                            color_single = color_dict['1']
                        elif names[int(cls)] == 'point':
                            color_single = color_dict['2']

                        ##annotator.box_label(xyxy, label, color=color_single
                        annotator.box_label(xyxy, label, color=colors(c, True)) #调用annotator这个类的box_label这个函数
                    if save_crop: #判断是否保存截下来的目标框，默认为false
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result() #im0：从annotator中返回画好框的图片
            if view_img:#如果view_img为true的话，把画好框的图片显示一下，显示成一个窗口，展示一下，如果为false，不显示
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img: #如果想把图片保存下来
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0) #cv2.imwrite open CV保存图片的函数
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results #第6部分：最终打印出输出信息
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # #统计每张图片的平均时间，seen记录了总共预测了多少张图片，dt总共的耗时，预处理，推断，非极大值抑制各自的耗时，打印信息的speed
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:#如果结果保存为txt或img，打印信息results save to ...
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():#解析命令行参数的函数：3个部分
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')#定义些命令行可以传入的参数，权重
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')#模型预测的图片大小
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand # #如果imgsz=640，640，返回1，否则640*2即640，640
    print_args(vars(opt)) #将所有的参数信息打印出来，运行结果detect：后面的内容。opt变量用来存储所有的参数信息
    return opt #变量返回，之后执行main（opt）函数


def main(opt):#main函数执行了2个函数
    check_requirements(exclude=('tensorboard', 'thop')) #函数1：检测python.txt所需的依赖包是否安装
    run(**vars(opt)) #函数2：检测完成后，执行run函数，把opt参数传进去，run函数就是后续一系列图片加载，预测及结果保存等。向上找到run函数


if __name__ == '__main__': #倒包结束后直接跳到这里执行if main代码
    opt = parse_opt() #函数1：解析命令行传入的参数 eg.--source data\\images\\bus.jpg
    main(opt) #函数2：执行自己定义的main函数，把opt变量传给main函数
