import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import easyocr
import os
os.environ['USE_TORCH']='1'
import matplotlib as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import tkinter as tk
from PIL import Image,ImageTk
from tkinter import messagebox

import pytesseract as pyt
import pymysql
from datetime import datetime
import shutil

from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def plaka_bul():

    def detect(save_img=False):
        source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        #half = device.type != 'cpu'  # half precision only supported on CUDA
        half=False

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, opt.img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()
            print("Predictions (after NMS):")
            if len(pred[0]) > 0:  # Pred listesi boş değilse devam et
                
                det_0 = pred[0][0][0]  # Pred'in 0. indeksindeki tensörü seç
                print(pred)
                print("Information at index 0:")
                print(type(det_0))
                print(det_0)
                print(det_0.item())
                det_0 = pred[0][0][1]
                print(det_0.item())
                det_0 = pred[0][0][2]
                print(det_0.item())
                det_0 = pred[0][0][3]
                print(det_0.item())
                konum=[]
                konum.append(pred[0][0][0].item())
                konum.append(pred[0][0][1].item())
                konum.append(pred[0][0][2].item())
                konum.append(pred[0][0][3].item())

                print("burada")
                print(path)
                resim=cv2.imread(path)
                #cut_img=resim[round(konum[0]):round(konum[2]),round(konum[1]):round(konum[3])]
                x=round(konum[0])#w1
                y=round(konum[1])#h1
                w=round(konum[2])#w2
                h=round(konum[3])#h2

                cut_img=resim[y:h,x:w]

                #print(round(konum[0]), round(konum[1]), round(konum[2]), round(konum[3]))
                kesilen_resim=cv2.imshow("cut image",cut_img)
                cv2.imwrite("C:/Users/yenie/Desktop/carplate/yolov7/kaydedilen_resimler/resim.jpg",cut_img)
                cv2.waitKey(0)
                if len(pred[0]) > 1:  # Pred listesi boş değilse devam et
                
                    det_0 = pred[0][1][0]  # Pred'in 0. indeksindeki tensörü seç
                    print(pred)
                    print("Information at index 0:")
                    print(type(det_0))
                    print(det_0)
                    print(det_0.item())
                    det_0 = pred[0][1][1]
                    print(det_0.item())
                    det_0 = pred[0][1][2]
                    print(det_0.item())
                    det_0 = pred[0][1][3]
                    print(det_0.item())
                    konum=[]
                    konum.append(pred[0][1][0].item())
                    konum.append(pred[0][1][1].item())
                    konum.append(pred[0][1][2].item())
                    konum.append(pred[0][1][3].item())

                    print("burada")
                    print(path)
                    resim=cv2.imread(path)
                    #cut_img=resim[round(konum[0]):round(konum[2]),round(konum[1]):round(konum[3])]
                    x=round(konum[0])#w1
                    y=round(konum[1])#h1
                    w=round(konum[2])#w2
                    h=round(konum[3])#h2

                    cut_img=resim[y:h,x:w]

                    #print(round(konum[0]), round(konum[1]), round(konum[2]), round(konum[3]))
                    kesilen_resim=cv2.imshow("cut image",cut_img)
                    cv2.imwrite("C:/Users/yenie/Desktop/carplate/yolov7/kaydedilen_resimler/resim.jpg",cut_img)
                    cv2.waitKey(0)
            else:
                print("Herhangi bir plaka tespit edilemedi")

            #for i, det in enumerate(pred):
            #    print(f"Image {i}: {det}")
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    #cv2.imshow(str(p), im0)
                    #cv2.waitKey(1000)  # 1 millisecond
                    break

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        #cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='C:/Users/yenie/Desktop/yolov7_sonuclari/bestson.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='cekilen_resimler', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        opt = parser.parse_args()
        print(opt)
        #check_requirements(exclude=('pycocotools', 'thop'))

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov7.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                detect()


    path= "C:/Users/yenie/Desktop/carplate/yolov7/kaydedilen_resimler/resim.jpg"

    pic= cv2.imread(path)

    scale=4
    width=int(pic.shape[1]*scale)
    height=int(pic.shape[0]*scale)
    pic_resized=cv2.resize(pic,(width,height)) 

    #solu kırp

    pic_cropped=pic_resized[0:pic_resized.shape[0],int(pic_resized.shape[1]*0.047):pic_resized.shape[1]]

    filtered_image=cv2.bilateralFilter(pic_cropped,11,17,17)    #bilareteral filtre 25,100,100


    sharp_filter=np.array([
        [0,-1,0],
        [-1,4.96,-1],         #ortadaki değerin kameranın arabalara uzaklığına göre optimize edilmesi gerekiyor. keskinleştirme filtresi
        [0,-1,0]
    ])
    image=cv2.filter2D(filtered_image,ddepth=-1,kernel=sharp_filter)


    pic= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #BGR2GRAY

    imw=cv2.threshold(pic,None,200,cv2.THRESH_OTSU)[1]  #thresholdlar arasındaki farkları gösterebilrsin

    #cannyedge=cv2.Canny(image=imw,threshold1=127,threshold2=127)

    #filtered_image=cv2.bilateralFilter(imw,25,100,100)  #bilareteral filtre

    cv2.imwrite(path,imw)


    #easyocr
    reader=easyocr.Reader(['en'],gpu=True)
    result=reader.readtext(path)
    print(result)    


    image=cv2.imread(path)

    #docTR
    model=ocr_predictor(pretrained=True)
    Document=DocumentFile.from_images(path)
    result=model(Document)
    print(result)

    #tesseract
    custom_config = r'--oem 3 --psm 6'
    pyt.pytesseract.tesseract_cmd= "C:/Program Files/Tesseract-OCR/tesseract.exe"
    result=pyt.image_to_string(image, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVYZ0123456789')
    print(result)

    if len(result) >= 3:
        if result[1].isdigit() and result[2].isdigit():
            result=result[1:]
    print(result)

    sql="SELECT * FROM plaka WHERE plaka= %s AND cikis IS NULL LIMIT 1"
    cursor.execute(sql,(result))
    sql_result=cursor.fetchone()
    zaman=int(time.time())

    if sql_result:
        sql="SELECT giris FROM plaka WHERE plaka= %s AND cikis IS NULL LIMIT 1"
        cursor.execute(sql,(result))
        giris=cursor.fetchone()
        giris=int(giris[0])
        ucret=zaman-giris

        sql_update="UPDATE plaka SET cikis = %s WHERE plaka = %s AND cikis IS NULL"
        cursor.execute(sql_update,(zaman,result))
        baglanti.commit()

        messagebox.showinfo("Ücret","Ücret: "+str(ucret)+" TL")
    else:
        sql="INSERT INTO plaka (plaka,giris) VALUES (%s,%s)"
        cursor.execute(sql,(result,zaman))
        baglanti.commit()
        messagebox.showinfo("Başarılı", "Araç başarıyla giriş yaptı!")
    
    return result

baglanti=pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='otopark'
)
cursor=baglanti.cursor()

img_counter=0
def take_photo():
    
    folder_path="cekilen_resimler"
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    global img_counter
    ret, frame = cap.read()
    if ret:
        img_name="cekilen_resimler/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name,frame)
        img_counter+=1
        #cv2.imwrite("C:/Users/yenie/Desktop/foto/photo.jpg", frame)
        messagebox.showinfo("Başarılı", "Fotoğraf çekildi ve kaydedildi!")
        plaka=plaka_bul()
        update_listbox()
        #listbox.insert("end",plaka)
    else:
        messagebox.showerror("Hata", "Fotoğraf çekilemedi!")

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Kamera açılamadı!")

root=tk.Tk()
root.title("Otopark Uygulaması")

label= tk.Label(root, text="Otoparktaki Araçlar")
label.pack()

#liste
listbox = tk.Listbox(root)
listbox.pack()

def update_listbox():
    listbox.delete(0,"end")
    sql="SELECT plaka FROM plaka WHERE cikis IS NULL"
    cursor.execute(sql)
    plakalar=cursor.fetchall()
    
    for plaka in plakalar:
        listbox.insert("end",plaka)
    
update_listbox() 


def update_frame():
    ret,frame=cap.read()
    if ret:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img=Image.fromarray(frame)
        imgtk=ImageTk.PhotoImage(image=img)
        lmain.imgtk=imgtk
        lmain.configure(image=imgtk)
    lmain.after(10, update_frame)

lmain=tk.Label(root)
lmain.pack()

btn=tk.Button(root, text="Fotoğraf Çek", command=take_photo)
btn.pack()

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()

"""
    cam= cv2.VideoCapture(0)
    cv2.namedWindow("Resmini Çek")
    img_counter=0

    while(True):
        ret,frame=cam.read()
        
        if not ret:
            print("hata")
            break
        
        cv2.imshow("test",frame)
        e=cv2.waitKey(1)

        if e%256==27:
            print("uygulama kapanıyor")
            break
        
        elif e%256==32:
            img_name="cekilen_resimler/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name,frame)
            print("resim çekildi")
            img_counter+=1

    cam.release()
    cv2.destroyAllWindows()"""





"""
arayuz=tk.Tk()
arayuz.title("Otopark Uygulaması")
yazi=tk.Label(arayuz,text="yazı")
yazi.pack()

arayuz.geometry('500x250')
arayuz.resizable(False,False)

def topla():
    print(100)
buton=tk.Button(arayuz,text="tıkla",command=topla)
buton.pack()

arayuz.mainloop()"""
"""
class webcam:
    def __init__(self,window):
        self.window=window
        self.window.title("Otopark uygulaması")

        self.video_capture=cv2.VideoCapture(2)
        self.current_image=None
        self.canvas=tk.Canvas(window,width=640,height=480)
        self.canvas.pack()

arayuz=tk.Tk()

uygulama=webcam(arayuz)

arayuz.mainloop()
"""