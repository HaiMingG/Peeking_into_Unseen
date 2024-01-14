import torch
from configs import data_dir, categories
from configs import *
from pycocotools.coco import COCO
from PIL import Image, ImageOps
from torchvision import transforms
import pycocotools.mask as maskUtils
from util import *
import json

def resize_bbox(img, bboxs, short=224, single=False, interp=False):
    h, w  = img.shape[0:2]

    if single:
        box = bboxs
        factor = short / min(box[2] - box[0], box[3] - box[1])
    else:
        short_side = []
        for box in bboxs:
            short_side.append(min(box[2] - box[0], box[3] - box[1]))
        factor = short / np.min(short_side)

    resized_bboxs = bboxs * factor
    if interp:
        resized_img = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation = cv2.INTER_NEAREST)
    else:
        resized_img = cv2.resize(img, (int(w * factor), int(h * factor)))
    return resized_img, resized_bboxs

def resize_scale(img, scale=1, interp=False):
    h, w  = img.shape[0:2]
    factor = scale

    if interp:
        resized_img = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation = cv2.INTER_NEAREST)
    else:
        resized_img = cv2.resize(img, (int(w * factor), int(h * factor)))
    return resized_img


class Single_Object_Loader():
    def __init__(self, image_files, mask_files, labels, bboxs, resize=True, ss_length=224, crop_img=True, crop_padding=48, crop_central=False, demo_img_return=True, return_true_pad=False):
        self.image_files = image_files
        self.mask_files = mask_files
        self.labels = labels
        self.bboxs = bboxs

        self.resize_bool = resize                   #boolean:   resize image
        self.resize_side = ss_length                #int:       resize side length
        self.crop_bool = crop_img                   #boolean:   crop image
        self.crop_pad = crop_padding                #int:       crop padding
        self.crop_central = crop_central            #boolean:   same padding on all 4 sides
        self.demo_bool = demo_img_return            #boolean:   return demo image corresponding to the float tensor
        self.return_true_pad = return_true_pad      #boolean:   return true pad length

    def __getitem__(self, index):

        img_path = self.image_files[index]
        mask_path = self.mask_files[index]
        label = self.labels[index]
        bbox = self.bboxs[index]

        input_image = Image.open(img_path)
        sz = input_image.size                   # W, H

        mask = np.ones((sz[1], sz[0], 3))
        if os.path.exists(mask_path):
            annotation = np.load(mask_path)
            mask[:, :, 0] = annotation['mask']

        demo_img = []

        if self.demo_bool:
            demo_img = cv2.imread(img_path)


        if self.resize_bool:
            short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if short_side < 3:
                print('Bad Bbox Annotation:', index, img_path, bbox)
                bbox = np.array([0, 0, sz[1], sz[0]])
                short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])

            input_image = input_image.resize((np.asarray(sz) * (self.resize_side / short_side)).astype(int), Image.ANTIALIAS)
            sz = input_image.size
            mask, _ = resize_bbox(mask, bbox, single=True, interp=True, short=self.resize_side)

            if self.demo_bool:
                demo_img, _ = resize_bbox(demo_img, bbox, single=True, short=self.resize_side)

            bbox = (bbox * (self.resize_side / short_side)).astype(int)

        pad = self.crop_pad
        if self.crop_bool:

            box = bbox

            if self.crop_central:
                box[0] = max(box[0], 0)
                box[1] = max(box[1], 0)
                box[2] = min(box[2], sz[1])
                box[3] = min(box[3], sz[0])
                pad = min(box[0] - 0, box[1] - 0, sz[1] - box[2], sz[0] - box[3], self.crop_pad)

            left = max(0, box[1] - pad)
            top = max(0, box[0] - pad)
            right = min(sz[0], box[3] + pad)
            bottom = min(sz[1], box[2] + pad)

            input_image = input_image.crop((left, top, right, bottom))
            mask = (mask[top:bottom, left:right, 0] > 127).astype(float)

            if self.demo_bool:
                demo_img = demo_img[top:bottom, left:right, :]

        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        if np.sum(mask) == 0:
            mask = 1 - mask

        if self.return_true_pad:
            return input_tensor, label, bbox, mask, demo_img, img_path, pad

        return input_tensor, label, bbox, mask, demo_img, img_path

    def __len__(self):
        return len(self.image_files)


class Multi_Object_Loader():
    def __init__(self, image_files, mask_files, labels, bboxs, resize=True, min_size=99999, max_size=0, demo_img_return=True):
        self.image_files = image_files
        self.mask_files = mask_files
        self.labels = labels
        self.bboxs = bboxs

        self.resize_bool = resize                   #boolean:   resize image
        self.demo_bool = demo_img_return            #boolean:   return demo image corresponding to the float tensor
        self.max_size = max_size
        self.min_size = min_size

    def __getitem__(self, index):

        img_path = self.image_files[index]
        mask_path = self.mask_files[index]
        label = self.labels[index]
        bbox = self.bboxs[index]

        input_image = Image.open(img_path)
        sz = input_image.size

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
        else:
            mask = np.zeros((sz[1], sz[0], 3))

        demo_img = []

        if self.demo_bool:
            demo_img = cv2.imread(img_path)


        if self.resize_bool:
            box = bbox[0]
            short_side = min(box[2] - box[0], box[3] - box[1])
            if short_side < 3:
                bbox = np.array([[0, 0, sz[1], sz[0]]])
                box = bbox[0]
                short_side = min(box[2] - box[0], box[3] - box[1])

            input_image = input_image.resize((np.asarray(sz) * (224 / short_side)).astype(int), Image.ANTIALIAS)
            sz = input_image.size
            mask, _ = resize_bbox(mask, box, single=True, interp=True)

            if self.demo_bool:
                demo_img, _ = resize_bbox(demo_img, box, single=True)

            bbox = (bbox * (224 / short_side)).astype(int)

        if sz[0] > self.max_size or sz[1] > self.max_size:
            scale = self.max_size / max(sz[0], sz[1])
            input_image = input_image.resize((np.asarray(sz) * scale).astype(int), Image.ANTIALIAS)
            mask = resize_scale(mask, scale=scale, interp=False)
            if self.demo_bool:
                demo_img = resize_scale(demo_img, scale=scale, interp=False)

            bbox = (bbox * scale).astype(int)
        else:
            scale = 1.

        if sz[0] < self.min_size or sz[1] < self.min_size:
            scale = self.min_size / min(sz[0], sz[1])
            input_image = input_image.resize((np.asarray(sz) * scale).astype(int), Image.ANTIALIAS)
            mask = resize_scale(mask, scale=scale, interp=False)
            if self.demo_bool:
                demo_img = resize_scale(demo_img, scale=scale, interp=False)

            bbox = (bbox * scale).astype(int)
        else:
            scale = 1.

        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        return input_tensor, label, bbox, mask, scale, demo_img, img_path

    def __len__(self):
        return len(self.image_files)



#Major Dataset for this project, includes both inmodal and amodal segmentation mask for analysis
class KINS_Dataset():
    def __init__(self, category_list, dataType='train', occ=(0,1), height_thrd=50, amodal_height=True, data_range=(0, 1), demo_img_return=True):

        self.src_data_path = data_dir + 'kitti/{}ing/'.format(dataType)
        self.image_ids = []
        self.obj_ids = []
        self.demo_bool = demo_img_return
        occ_lb = occ[0]
        occ_ub = occ[1]
        assert occ_lb <= occ_ub

        cat_kins = []
        cat_kins.append(categories['kins'].index('pig'))

        filelist = '{}list.txt'.format(self.src_data_path)

        with open(filelist, 'r') as fh:
            contents = fh.readlines()
        fh.close()
        img_list = [cc.strip() for cc in contents]
        N = len(img_list)
        img_list = img_list[int(data_range[0] * N):int(data_range[1] * N)]
        for ii, img_id in enumerate(img_list):

            if ii % 10 == 0:
                print('Loading Data: {}/{}'.format(ii, N), end='\r')

            annotation = np.load('{}annotations/{}.npz'.format(self.src_data_path, img_id))
            obj_ids = annotation['obj_ids']
            labels = annotation['labels']
            occlusion_fractions = annotation['occluded_percentage']
            if amodal_height:
                bboxes = annotation['amodal_bbox']  # dim = (N, 4) --> (y1, x1, y2, x2)
            else:
                bboxes = annotation['inmodal_bbox']


            obj_ids_per_img = []

            for i in range(obj_ids.shape[0]):
                box = bboxes[i]           #inmodal_bboxes, amodal_bboxes

                if labels[i] in cat_kins and occlusion_fractions[i] >= occ_lb and occlusion_fractions[i] <= occ_ub and box[3] - box[1] >= height_thrd:
                    obj_ids_per_img.append(obj_ids[i])

            if len(obj_ids_per_img) > 0:
                self.image_ids.append(img_id)
                self.obj_ids.append(obj_ids_per_img)


    def __getitem__(self, index):

        img_id = self.image_ids[index]
        obj_id = self.obj_ids[index]

        img_path = self.src_data_path + 'images/{}.png'.format(img_id)

        input_image = Image.open(img_path)
        demo_img = []
        if self.demo_bool:
            demo_img = cv2.imread(img_path)

        annotation = np.load('{}annotations/{}.npz'.format(self.src_data_path, img_id), allow_pickle=True)

        obj_ids = annotation['obj_ids']
        inmodal_bbox = annotation['inmodal_bbox']
        amodal_bbox = annotation['amodal_bbox']

        labels = annotation['labels']
        occlusion_fractions = annotation['occluded_percentage']

        # dim = [ encode, encode, encode ]
        inmodal_masks_ = annotation['inmodal_mask']
        amodal_masks_ = annotation['amodal_mask']

        gt_inmodal_bbox = []
        gt_amodal_bbox = []
        gt_labels = []
        gt_occ = []
        gt_inmodal_segentation = []
        gt_amodal_segentation = []

        for id in obj_id:
            index = np.where(obj_ids == id)[0][0]

            box = inmodal_bbox[index]
            gt_inmodal_bbox.append(np.array([box[1], box[0], box[3], box[2]]))

            box = amodal_bbox[index]
            gt_amodal_bbox.append(np.array([box[1], box[0], box[3], box[2]]))

            if categories['kins'][labels[index]] in categories['train']:
                gt_labels.append( categories['train'].index( categories['kins'][labels[index]] ) )
            else:
                gt_labels.append(-1)

            gt_occ.append(occlusion_fractions[index])
            gt_inmodal_segentation.append(maskUtils.decode(inmodal_masks_[index])[:, :, np.newaxis].squeeze())
            gt_amodal_segentation.append(maskUtils.decode(amodal_masks_[index][0]).squeeze())

        gt_inmodal_bbox = np.array(gt_inmodal_bbox)
        gt_amodal_bbox = np.array(gt_amodal_bbox)
        gt_labels = np.array(gt_labels)
        gt_occ = np.array(gt_occ)
        gt_inmodal_segentation = np.array(gt_inmodal_segentation)
        gt_amodal_segentation = np.array(gt_amodal_segentation)

        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        return input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segentation, gt_amodal_segentation, gt_occ, demo_img, img_path, False


    def __len__(self):
        return len(self.image_ids)

# Dataset based on the PASCAL3D+ Dataset with artificial generated occlusions -- Currently Active

class KINS_Compnet_Train_Dataset():
    def __init__(self, height_thrd=50, frac=1.0, pad=0, demo_img_return=True):
        self.src_data_path = data_dir + 'kitti/compnet_training/'

        self.files = []
        self.bbox = []
        self.demo_bool = demo_img_return

        with open('{}_annotations.pickle'.format(self.src_data_path), 'rb') as fh:
            annotations = pickle.load(fh)
        fh.close()

        for anno in annotations:
            if anno['short_side'] >= height_thrd:
                if pad <= 0:
                    self.files.append(anno['file'])
                else:
                    self.files.append(anno['org_file'])
                    self.bbox.append(anno['inmodal_bbox'])

        self.files = self.files[0:int(frac * len(self.files))]
        self.pad = pad
        np.random.seed(0)
        self.random_transform = np.random.permutation(len(self.files))

        if pad > 0:
            self.src_data_path = data_dir + 'kitti/training/'

    def __getitem__(self, index):
        index = self.random_transform[index]
        img_path = self.src_data_path + 'images/{}.png'.format(self.files[index])
        input_image = Image.open(img_path)
        pad = 0

        if self.pad > 0:
            bbox = self.bbox[index]

            factor = 224 / (bbox[2] - bbox[0])

            input_image = input_image.resize((np.asarray(input_image.size) * factor).astype(int))
            bbox = (np.array(bbox) * factor).astype(int)
            sz = input_image.size
            pad = min(bbox[0] - 0, bbox[1] - 0, sz[1] - bbox[2], sz[0] - bbox[3], self.pad)

            top = max(0, bbox[0] - pad)
            left = max(0, bbox[1] - pad)
            bottom = min(sz[1], bbox[2] + pad)
            right = min(sz[0], bbox[3] + pad)

            input_image = input_image.crop((left, top, right, bottom))

        demo_img = []
        if self.demo_bool:
            demo_img = np.asarray(input_image)

        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        return input_tensor, demo_img, img_path, pad


    def __len__(self):
        return len(self.files)
