import math
import copy
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.modules.utils import _pair, _quadruple

from configs import device_ids, feat_stride, dataset_eval
from configs import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Net(nn.Module):

    def __init__(self, Feature_Extractor, VC_Centers, Context_Kernels, Mixture_Models, Clutter_Models, vMF_kappa, omega=0.2):
        super(Net, self).__init__()

        self.extractor = Feature_Extractor

        self.vc_conv1o1 = Conv1o1Layer(VC_Centers)
        self.context_conv1o1 = Conv1o1Layer(Context_Kernels)

        fg_models, fg_priors, context_models, context_priors = Mixture_Models

        self.pig_fg_model = fg_models
        self.pig_fg_prior = fg_priors
        self.pig_context_model = context_models
        self.pig_context_prior = context_priors

        self.fg_models = [self.pig_fg_model]
        self.fg_prior = [self.pig_fg_prior]
        self.context_models = [self.pig_context_model]
        self.context_prior = [self.pig_context_prior]

        self.clutter_conv1o1 = Conv1o1Layer(Clutter_Models)
        self.exp = ExpLayer(vMF_kappa)
        self.median_filter = MedianPool2d(kernel_size=3, stride=1, same=True)
        self.soft_max = SoftMax(2)
        self.relu = nn.ReLU()

        self.kappa = vMF_kappa
        self.num_class = len(self.fg_models)
        self.fused_models = []
        self.omega = omega
        self.train = False

        standard_filter = [[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]], [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]]

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = torch.nn.Conv2d(2, 2, 3, bias=True, padding=(1, 1))
        self.conv1.weight = torch.nn.Parameter(torch.tensor(standard_filter))
        self.conv1.bias = torch.nn.Parameter(torch.tensor([0., 0.]))

        self.conv2 = torch.nn.Conv2d(2, 2, 3, bias=True, padding=(1, 1))
        self.conv2.weight = torch.nn.Parameter(torch.tensor(standard_filter))
        self.conv2.bias = torch.nn.Parameter(torch.tensor([0., 0.]))

        self.conv3 = torch.nn.Conv2d(2, 2, 3, bias=True, padding=(1, 1))
        self.conv3.weight = torch.nn.Parameter(torch.tensor(standard_filter))
        self.conv3.bias = torch.nn.Parameter(torch.tensor([0., 0.]))

        self.conv4 = torch.nn.Conv2d(2, 2, 3, bias=True, padding=(1, 1))
        self.conv4.weight = torch.nn.Parameter(torch.tensor(standard_filter))
        self.conv4.bias = torch.nn.Parameter(torch.tensor([0., 0.]))

        self.conv5 = torch.nn.Conv2d(2, 2, 3, bias=True, padding=(1, 1))
        self.conv5.weight = torch.nn.Parameter(torch.tensor(standard_filter))
        self.conv5.bias = torch.nn.Parameter(torch.tensor([0., 0.]))



    # ====================== Functional Methods ======================#

    def forward(self, org_x, bboxes, bbox_type='amodal', input_label=None, gt_mask_label=False, mask_label_training=None, crop_pad=32, slide_window_stride=2, only_cls=False):

        score, center, pred_mixture, pred_amodal_bboxes = self.classify(org_x, bboxes=bboxes, pad_length=crop_pad, slide_window=(bbox_type=='inmodal'),
                                                                  stride=slide_window_stride, gt_labels=input_label)
        pred_labels = score.argmax(1)

        if only_cls:
            return score, self.soft_max(score), None, None

        if gt_mask_label:
            _score_, _, pred_mixture, pred_amodal_bboxes = self.classify(org_x, bboxes=bboxes, pad_length=crop_pad, slide_window=(bbox_type == 'inmodal'), stride=slide_window_stride, gt_labels=mask_label_training)
            pred_labels = _score_.argmax(1)

        amodal_bboxes = bboxes

        pred_segmentations = self.segment(org_x, bboxes=amodal_bboxes, labels=pred_labels, mixture=pred_mixture, pad_length=crop_pad)

        return score, self.soft_max(score), pred_amodal_bboxes, pred_segmentations

    def classify(self, org_x, bboxes, pad_length=16, slide_window=True, stride=1, gt_labels=None):

        score = -1 * torch.ones((bboxes.shape[0], len(self.fg_models))).cuda(device_ids[0])
        center = -1 * torch.ones((bboxes.shape[0], len(self.fg_models), 2)).cuda(device_ids[0])
        mixture = -1 * torch.ones((bboxes.shape[0], len(self.fg_models))).cuda(device_ids[0])
        amodal_bboxes = -1 * torch.ones((bboxes.shape[0], 4)).cuda(device_ids[0])

        amodal_comp = torch.zeros((bboxes.shape[0], len(self.fg_models))).cuda(device_ids[0])
        for ii, box in enumerate(bboxes):

            if (box[3] - box[1]) / (box[2] - box[0]) > 30:
                box = torch.tensor([0, 0, org_x.shape[2], org_x.shape[3]])

            # calculate rescale factors
            factor = 224 / float(box[2] - box[0])

            # pad and crop ground truth box
            pad = max(0, min(int(pad_length / factor), box[0] - 0, box[1] - 0, org_x.shape[2] - box[2], org_x.shape[3] - box[3]))
            x = org_x[:, :, box[0] - pad: box[2] + pad, box[1] - pad: box[3] + pad]
            # rescale padded box to height 224
            x = F.interpolate(x, scale_factor=factor, recompute_scale_factor=True)

            # Obtain vc activation
            x = self.backbone(x)

            # Obtain outlier activation
            clutter_match = self.log(self.clutter_conv1o1(x, norm=False))
            clutter_match, _ = clutter_match.max(1)

            try:
                cat_to_eval = [gt_labels[ii]]
            except:
                cat_to_eval = [i for i in range(len(self.fg_models))]

            for category in cat_to_eval:
                model_h, model_w = self.fused_models[category].shape[2:]

                if slide_window:
                    best_val, best_center, best_k = self.slide_window_on_model(x, clutter_match, category=category, stride=stride)

                    best_k_, best_val_ = self.get_best_mixture(x, clutter_match, category=category)
                    if best_val_ >= best_val:
                        center[ii][category][0] = int(model_h / 2)
                        center[ii][category][1] = int(model_w / 2)
                    else:
                        center[ii][category][0] = best_center[0].item()
                        center[ii][category][1] = best_center[1].item()
                        amodal_comp[ii][category] = 1
                else:

                    best_k, best_val = self.get_best_mixture(x, clutter_match, category=category)
                    center[ii][category][0] = int(model_h / 2)
                    center[ii][category][1] = int(model_w / 2)

                score[ii][category] = best_val / (x.shape[2] * x.shape[3])
                mixture[ii][category] = best_k

            # Estimate amodal box
            pred_label = torch.argmax(score[ii]).item()
            pred_center = center[ii][pred_label]
            pred_mixture = mixture[ii][pred_label]

            if amodal_comp[ii][pred_label] == 1:
                amodal_bboxes[ii] = self.amodal_box_completion(x, partial_box=box, clutter_match=clutter_match,
                                                               pred_label_center_mixture=[pred_label, pred_center,
                                                                                          pred_mixture], factor=factor,
                                                               image_size=org_x.shape[2:])
            else:
                amodal_bboxes[ii] = box.cuda(device_ids[0])

        return score, center, mixture.type(torch.LongTensor), amodal_bboxes.type(torch.LongTensor)

    def segment(self, org_x, bboxes, labels, mixture, pad_length=16):

        segmentations = []

        for ii, box in enumerate(bboxes):

            ###    Pre-process image patch
            if box[2] - box[0] < 10:
                box[0] = 0
                box[2] = org_x.shape[2]
            patch_center = np.array([(box[2] + box[0]) / 2, (box[3] + box[1]) / 2])
            category = labels[ii]
            k_max = mixture[ii][category]

            factor = 224 / float(box[2] - box[0])
            restore_factor = feat_stride / factor

            pad = max(0, min(int(pad_length / factor), box[0] - 0, box[1] - 0, org_x.shape[2] - box[2], org_x.shape[3] - box[3]))
            x = org_x[:, :, box[0] - pad: box[2] + pad, box[1] - pad: box[3] + pad]
            x = F.interpolate(x, scale_factor=factor, recompute_scale_factor=True)
            x = self.backbone(x)

            fg_prior = self.center_crop(self.fg_prior[category], x.shape[2:])
            _, model_h, model_w = fg_prior.shape


            pixel_cls_score, pixel_cls = self.segment_per_pixel_cls(x, category=category, k_max=k_max)

            object_seg = dict()

            object_seg['pixel_cls'] = self.place_mask_in_image(mask=pixel_cls.cpu().detach().numpy().squeeze(), mask_dim_in_img=[model_h * restore_factor, model_w * restore_factor],
                                                        center_in_img=patch_center, org_img_shape=org_x.shape[2:4])

            object_seg['pixel_cls_score'] = self.place_mask_in_image(mask=pixel_cls_score.cpu().detach().numpy().squeeze(), mask_dim_in_img=[model_h * restore_factor, model_w * restore_factor],
                                                               center_in_img=patch_center, org_img_shape=org_x.shape[2:4])


            binary_resp = self.binarize_pixel_cls(pixel_cls, pixel_cls_score, target_label=1, rigid_prior=True)
            if binary_resp.shape[2] > 2:
                binary_resp = self.median_filter(binary_resp.unsqueeze(0))[0]

            inmodal_mask = self.interpolate_mask(primary_resp=binary_resp * (binary_resp > 0).type(torch.FloatTensor).cuda(device_ids[0]), coounter_resp=binary_resp * (binary_resp < 0).type(torch.FloatTensor).cuda(device_ids[0]) * -1)
            object_seg['inmodal_raw'] = inmodal_mask

            inmodal_mask_in_img = self.place_mask_in_image(mask=inmodal_mask[0][1].cpu().detach().numpy().squeeze(), mask_dim_in_img=[model_h * restore_factor, model_w * restore_factor],
                                                          center_in_img=patch_center, org_img_shape=org_x.shape[2:4])

            object_seg['inmodal'] = np.zeros(org_x.shape[2:])
            object_seg['inmodal'][box[0]:box[2], box[1]:box[3]] = inmodal_mask_in_img[box[0]:box[2], box[1]:box[3]]



            self.add_fg_prior(pixel_cls, pixel_cls_score, fg_prior[k_max].cpu(), thrd=0.9)
            binary_resp = self.binarize_pixel_cls(pixel_cls, pixel_cls_score, target_label=1, rigid_prior=True)
            if binary_resp.shape[2] > 2:
                binary_resp = self.median_filter(binary_resp.unsqueeze(0))[0]

            amodal_mask = self.interpolate_mask( primary_resp=binary_resp * (binary_resp > 0).type(torch.FloatTensor).cuda(device_ids[0]), coounter_resp=binary_resp * (binary_resp < 0).type(torch.FloatTensor).cuda(device_ids[0]) * -1)
            object_seg['amodal_raw'] = amodal_mask

            amodal_mask_in_img = self.place_mask_in_image(mask=amodal_mask[0][1].cpu().detach().numpy().squeeze(), mask_dim_in_img=[model_h * restore_factor, model_w * restore_factor], center_in_img=patch_center, org_img_shape=org_x.shape[2:4])

            object_seg['amodal'] = np.zeros(org_x.shape[2:])
            object_seg['amodal'][box[0]:box[2], box[1]:box[3]] = amodal_mask_in_img[box[0]:box[2], box[1]:box[3]]


            object_seg['occ'] = (object_seg['amodal'] - object_seg['inmodal']) * (object_seg['amodal'] - object_seg['inmodal'] >= 0).astype(float)

            segmentations.append(object_seg)
        return segmentations

    def interpolate_mask(self, primary_resp, coounter_resp):
        h, w = primary_resp.shape[-2:]
        x = torch.stack([coounter_resp, primary_resp]).view(1, 2, h, w).cuda(device_ids[0])
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.up(x)
        x = self.conv4(x)
        x = self.up(x)
        x = self.conv5(x)
        return self.soft_max(x)

    def place_mask_in_image(self, mask, mask_dim_in_img, center_in_img, org_img_shape, empty_fill=-1):
        segmentation = np.ones(org_img_shape) * np.min(mask)

        mask_as_img = cv2.resize(mask, (int(mask_dim_in_img[1]), int(mask_dim_in_img[0])), interpolation=cv2.INTER_NEAREST)

        mask_h, mask_w = mask_as_img.shape

        top_img, left_img, bottom_img, right_img = [math.floor(center_in_img[0] - mask_h / 2),
                                                    math.floor(center_in_img[1] - mask_w / 2),
                                                    math.floor(center_in_img[0] + mask_h / 2),
                                                    math.floor(center_in_img[1] + mask_w / 2)]
        top_mask, left_mask, bottom_mask, right_mask = [0, 0, mask_h, mask_w]

        if top_img < 0:
            top_mask += 0 - top_img
            top_img = 0

        if left_img < 0:
            left_mask += 0 - left_img
            left_img = 0

        if bottom_img > segmentation.shape[0]:
            bottom_mask += segmentation.shape[0] - bottom_img
            bottom_img = segmentation.shape[0]

        if right_img > segmentation.shape[1]:
            right_mask += segmentation.shape[1] - right_img
            right_img = segmentation.shape[1]

        if bottom_mask - top_mask != bottom_img - top_img:
            height = min(bottom_mask - top_mask, bottom_img - top_img)
            bottom_mask = top_mask + height
            bottom_img = top_img + height

        if right_mask - left_mask != right_img - left_img:
            width = min(right_mask - left_mask, right_img - left_img)
            right_mask = left_mask + width
            right_img = left_img + width

        segmentation[top_img:bottom_img, left_img:right_img] = mask_as_img[top_mask:bottom_mask, left_mask:right_mask]

        return segmentation

    def add_fg_prior(self, per_pixel_cls, pixel_cls_score, prior, thrd=0.5):
        per_pixel_cls += (per_pixel_cls == 0).type(torch.LongTensor).cuda(device_ids[0]) * (prior > thrd).type(torch.LongTensor).cuda(device_ids[0])
        # pixel_cls_score = (per_pixel_cls == 0).type(torch.FloatTensor) * prior * pixel_cls_score + (per_pixel_cls != 0).type(torch.FloatTensor) * pixel_cls_score

    def binarize_pixel_cls(self, pixel_cls, pixel_cls_score, target_label=1, rigid_prior=False):
        obj_resp = pixel_cls_score * (pixel_cls == target_label).type(torch.FloatTensor).cuda(device_ids[0])
        non_obj_resp = pixel_cls_score * (pixel_cls != target_label).type(torch.FloatTensor).cuda(device_ids[0])

        if rigid_prior:
            pixel_cls_ = copy.deepcopy(pixel_cls)
            self.binary_mask_post_process(pixel_cls_[0])
            non_obj_resp[(pixel_cls_ - pixel_cls > 0)] = 0

        return obj_resp - non_obj_resp

    def segment_per_pixel_cls(self, x, category, k_max):

        ###     Generate Foreground Activation
        fg_model = self.center_crop(self.fg_models[category], x.shape[2:])
        fg_prior = self.center_crop(self.fg_prior[category], x.shape[2:])
        fg_match = self.log(fg_prior[k_max] * (x * fg_model[k_max]).sum(1))

        ###     Generate Background Activation
        context_model = self.center_crop(self.context_models[category], x.shape[2:])
        context_prior = 1 - fg_prior
        bg_match = self.log(context_prior[k_max] * (x * context_model[k_max]).sum(1))

        ###     Generate Outlier Activation
        clutter_match = self.log(self.clutter_conv1o1(x, norm=False))
        clutter_match, _ = clutter_match.max(1)
        clutter_match = self.log(fg_prior[k_max] * torch.exp(clutter_match))

        stacked_match = torch.stack([clutter_match, fg_match, bg_match])    #clutter 0, fg 1, context 2
        ind = stacked_match.argmax(0)
        val, _ = stacked_match.max(0)
        return val, ind

    def slide_window_on_model(self, x, clutter_match, category, stride=1):

        x_h, x_w = x.shape[2:]

        fg_model = self.fg_models[category]
        fg_prior = self.fg_prior[category]
        context_model = self.context_models[category]
        context_prior = self.context_prior[category]

        fused_model = (1 - self.omega) * fg_model * fg_prior.unsqueeze(1) + self.omega * context_model * context_prior.unsqueeze(1)

        if x_h > fused_model.shape[2] or x_w > fused_model.shape[3]:
            crop_h = min(x.shape[2], fused_model.shape[2])
            crop_w = min(x.shape[3], fused_model.shape[3])
            x = self.center_crop(x, [crop_h, crop_w])                                       #### TODO: might cause a problem
            clutter_match = self.center_crop(clutter_match, [crop_h, crop_w])

        heat_map = torch.zeros((fused_model.shape[2] - x.shape[2] + 1, fused_model.shape[3] - x.shape[3] + 1)).cuda(device_ids[0])          #indices on the top left corner
        k_max_map = -1 * torch.ones((fused_model.shape[2] - x.shape[2] + 1, fused_model.shape[3] - x.shape[3] + 1)).cuda(device_ids[0])

        for h in range(0, fused_model.shape[2] - x.shape[2] + 1, stride):
            for w in range(0, fused_model.shape[3] - x.shape[3] + 1, stride):
                fg_match = self.log((x * fused_model[:, :, h:h + x.shape[2], w:w + x.shape[3]]).sum(1))

                not_occ = ((fg_match - clutter_match) > 0).type(torch.FloatTensor).cuda(device_ids[0])

                view_point_scores = (not_occ * (fg_match - clutter_match)).sum((1, 2))

                heat_map[h][w] = view_point_scores.max() / (x_h * x_w)
                k_max_map[h][w] = torch.argmax(view_point_scores).item()

        if stride > 1:
            thrd = torch.mean(heat_map[k_max_map >= 0])

            for h in range(0, fused_model.shape[2] - x.shape[2] + 1):
                for w in range(0, fused_model.shape[3] - x.shape[3] + 1):
                    if k_max_map[h][w] < 0 and heat_map[h - h % stride][w - w % stride] > thrd:
                        fg_match = self.log((x * fused_model[:, :, h:h + x.shape[2], w:w + x.shape[3]]).sum(1))

                        not_occ = ((fg_match - clutter_match) > 0).type(torch.FloatTensor).cuda(device_ids[0])

                        view_point_scores = (not_occ * (fg_match - clutter_match)).sum((1, 2))

                        heat_map[h][w] = view_point_scores.max() / (x_h * x_w)
                        k_max_map[h][w] = torch.argmax(view_point_scores).item()

        heat_map = heat_map
        best_val = heat_map.max()
        best_arg = heat_map.argmax().item()
        best_center = [best_arg // heat_map.shape[1] + int(x_h / 2), best_arg % heat_map.shape[1] + int(x_w / 2)]
        best_k = k_max_map[best_arg // heat_map.shape[1], best_arg % heat_map.shape[1]]

        return best_val, np.array(best_center), best_k

    def amodal_box_completion(self, x, partial_box, pred_label_center_mixture, clutter_match, factor, image_size):
        pred_label, pred_center, pred_mixture = pred_label_center_mixture

        fg_prior = self.fg_prior[pred_label][int(pred_mixture)]
        model_h, model_w = fg_prior.shape

        h = int(pred_center[0] - int(x.shape[2] / 2))
        w = int(pred_center[1] - int(x.shape[3] / 2))

        if x.shape[2] > model_h or x.shape[3] > model_w:
            crop_h = min(x.shape[2], model_h)
            crop_w = min(x.shape[3], model_w)
            x = self.center_crop(x, [crop_h, crop_w])
            clutter_match = self.center_crop(clutter_match, [crop_h, crop_w])

        pred_resp = torch.zeros((model_h, model_w)).cuda(device_ids[0])

        fg_match = self.log(fg_prior[h:h + x.shape[2], w:w + x.shape[3]] * (x * self.fg_models[pred_label][:, :, h:h + x.shape[2], w:w + x.shape[3]][int(pred_mixture)]).sum(1))
        bg_match = self.log((1 - fg_prior)[h:h + x.shape[2], w:w + x.shape[3]] * (x * self.context_models[pred_label][:, :, h:h + x.shape[2], w:w + x.shape[3]][int(pred_mixture)]).sum(1))
        clutter_match = self.log(fg_prior[h:h + x.shape[2], w:w + x.shape[3]] * torch.exp(clutter_match))

        context_b = (bg_match - clutter_match > 0).type(torch.FloatTensor).cuda(device_ids[0])
        exp_resp = torch.exp(fg_match - (clutter_match * (1 - context_b) + bg_match * context_b))

        pred_resp[h:h + x.shape[2], w:w + x.shape[3]] = exp_resp / (exp_resp + 1)

        explained = ((pred_resp > 0.5).type(torch.FloatTensor).cuda(device_ids[0]) * (fg_prior > 0.5).type(torch.FloatTensor).cuda(device_ids[0])).sum() / (fg_prior > 0.5).type(torch.FloatTensor).cuda(device_ids[0]).sum()


        if explained > 0.6:
            amodal_box = partial_box.cuda(device_ids[0])
        else:
            amodal_box = self.predict_amodal_box(box=partial_box, obj_center_on_model=pred_center, fg_prior=fg_prior.detach().cpu().numpy(), factor=factor, image_size=image_size)

        return amodal_box

    def predict_amodal_box(self, box, obj_center_on_model, fg_prior, factor, image_size):

        model_h, model_w = fg_prior.shape
        restore_factor = feat_stride / factor

        inmodal_ctr = np.array([(box[2] + box[0]) / 2, (box[3] + box[1]) / 2])
        inmodal_size = np.array([box[2] - box[0], box[3] - box[1]])

        off_set = np.array([model_h / 2 - obj_center_on_model[0], model_w / 2 - obj_center_on_model[1]]) * restore_factor

        true_ctr = off_set + inmodal_ctr

        relative_true_ctr = off_set + inmodal_size / 2

        pred_amodal_height = max(abs(relative_true_ctr[0] - 0), abs(relative_true_ctr[0] - inmodal_size[0])) * 2
        pred_amodal_width = max(abs(relative_true_ctr[1] - 0), abs(relative_true_ctr[1] - inmodal_size[1])) * 2

        visible_obj_x, visible_obj_y = np.where(fg_prior > 0.5)

        mean_amodal_height = (np.max(visible_obj_x) - np.min(visible_obj_x)) * restore_factor
        mean_amodal_width = (np.max(visible_obj_y) - np.min(visible_obj_y)) * restore_factor

        if pred_amodal_height < mean_amodal_height and pred_amodal_width < mean_amodal_width:
            pred_amodal_height = mean_amodal_height
            pred_amodal_width = mean_amodal_width

        tlx = max(true_ctr[0] - pred_amodal_height / 2, 0)
        tly = max(true_ctr[1] - pred_amodal_width / 2, 0)
        brx = min(tlx + pred_amodal_height, image_size[0])
        bry = min(tly + pred_amodal_width, image_size[1])

        amodal_box = torch.tensor([tlx, tly, brx, bry]).cuda(device_ids[0])
        return amodal_box




    # ====================== Initialization Methods ======================#

    def classify_init(self, org_x, bboxes, pad_length=16, slide_window=True, stride=1, gt_labels=None):


        score = -1 * torch.ones((bboxes.shape[0], len(self.fg_models))).cuda(device_ids[0])
        center = -1 * torch.ones((bboxes.shape[0], len(self.fg_models), 2)).cuda(device_ids[0])
        mixture = -1 * torch.ones((bboxes.shape[0], len(self.fg_models))).cuda(device_ids[0])
        amodal_bboxes = -1 * torch.ones((bboxes.shape[0], 4)).cuda(device_ids[0])

        amodal_comp = torch.zeros((bboxes.shape[0], len(self.fg_models))).cuda(device_ids[0])
        for ii, box in enumerate(bboxes):

            if (box[3] - box[1]) / (box[2] - box[0]) > 30:
                box = torch.tensor([0, 0, org_x.shape[2], org_x.shape[3]])

            # calculate rescale factors
            factor = 224 / float(box[2] - box[0])

            # pad and crop ground truth box
            pad = max(0, min(int(pad_length / factor), box[0] - 0, box[1] - 0, org_x.shape[2] - box[2], org_x.shape[3] - box[3]))
            x = org_x[:, :, box[0] - pad: box[2] + pad, box[1] - pad: box[3] + pad]
            # rescale padded box to height 224
            x = F.interpolate(x, scale_factor=factor, recompute_scale_factor=True)

            # Obtain vc activation
            feat = self.extractor(x)
            vc_act = self.vc_conv1o1(feat)
            x = self.exp(vc_act)

            # Obtain outlier activation
            clutter_match = self.log(self.clutter_conv1o1(x, norm=False))
            clutter_match, _ = clutter_match.max(1)

            try:
                cat_to_eval = [gt_labels[ii]]
            except:
                cat_to_eval = [i for i in range(len(self.fg_models))]

            for category in cat_to_eval:
                model_h, model_w = self.fused_models[category].shape[2:]

                if slide_window:
                    best_val, best_center, best_k = self.slide_window_on_model(x, clutter_match, category=category, stride=stride)

                    best_k_, best_val_ = self.get_best_mixture(x, clutter_match, category=category)
                    if best_val_ >= best_val:
                        center[ii][category][0] = int(model_h / 2)
                        center[ii][category][1] = int(model_w / 2)
                    else:
                        center[ii][category][0] = best_center[0].item()
                        center[ii][category][1] = best_center[1].item()
                        amodal_comp[ii][category] = 1
                else:

                    best_k, best_val = self.get_best_mixture(x, clutter_match, category=category)
                    center[ii][category][0] = int(model_h / 2)
                    center[ii][category][1] = int(model_w / 2)

                score[ii][category] = best_val / (x.shape[2] * x.shape[3])
                mixture[ii][category] = best_k

            # Estimate amodal box
            pred_label = torch.argmax(score[ii]).item()
            pred_center = center[ii][pred_label]
            pred_mixture = mixture[ii][pred_label]

            if amodal_comp[ii][pred_label] == 1:
                amodal_bboxes[ii] = self.amodal_box_completion(x, partial_box=box, clutter_match=clutter_match,
                                                               pred_label_center_mixture=[pred_label, pred_center,
                                                                                          pred_mixture], factor=factor,
                                                               image_size=org_x.shape[2:])
            else:
                amodal_bboxes[ii] = box.cuda(device_ids[0])

        return score, center, mixture.type(torch.LongTensor), amodal_bboxes.type(torch.LongTensor), (feat, vc_act, x)

    def backbone(self, x):
        x = self.extractor(x)
        x = self.vc_conv1o1(x)
        x = self.exp(x)
        return x

    def get_feature_activation(self, x, resize=False):
        if resize:
            factor = 224. / float(x.shape[2])
            x = F.interpolate(x, scale_factor=factor, recompute_scale_factor=True)
        x = self.extractor(x)
        return x

    def get_vc_activation(self, x, low_res=False, res_factor=0.25):

        with torch.no_grad():
            if low_res:
                org_size = x.shape
                x = F.interpolate(x, scale_factor=res_factor)
                x = F.interpolate(x, size=org_size[2:])
            x = self.backbone(x)
            return x.cpu().detach().numpy().squeeze()

    def get_vc_activation_with_binary_fg_mask(self, img, gt_category, use_context_center=False, use_mixture_model=False, context_thrd=None, mmodel_thrd=None, bmask_post_process=False, cntxt_pad=0, low_res=False, res_factor=0.25):

        assert use_context_center ^ use_mixture_model

        with torch.no_grad():

            if use_context_center:
                assert context_thrd != None

                x = self.extractor(img)

                context_resp = self.get_context_response(x, category=gt_category)

                binary_fg_mask = (context_resp <= context_thrd).type(torch.FloatTensor)
                if low_res:
                    org_size = img.shape
                    x = F.interpolate(img, scale_factor=res_factor)
                    x = F.interpolate(x, size=org_size[2:])
                    x = self.extractor(x)

                x = self.vc_conv1o1(x)
                x = self.exp(x)

            if use_mixture_model:
                assert mmodel_thrd != None

                if low_res:
                    org_size = img.shape
                    img = F.interpolate(img, scale_factor=res_factor)
                    img = F.interpolate(img, size=org_size[2:])

                x = self.extractor(img)

                x = self.vc_conv1o1(x)
                x = self.exp(x)

                fg_model = self.center_crop(self.fg_models[gt_category], x.shape[2:])
                fg_prior = self.center_crop(self.fg_prior[gt_category], x.shape[2:])
                context_model = self.center_crop(self.context_models[gt_category], x.shape[2:])
                context_prior = 1 - fg_prior

                clutter_match = self.log(self.clutter_conv1o1(x, norm=False))
                clutter_match = clutter_match[:, 0, :, :]

                k_max, _ = self.get_best_mixture(x, clutter_match, category=gt_category)

                fg_match = self.log(fg_prior[k_max] * (x * fg_model[k_max]).sum(1))
                bg_match = self.log(context_prior[k_max] * (x * context_model[k_max]).sum(1))
                clutter_match = torch.zeros(fg_match.shape).cuda(device_ids[0])

                context_b = (bg_match - clutter_match > 0).type(torch.FloatTensor).cuda(device_ids[0])
                baseline = clutter_match * (1 - context_b) + bg_match * context_b

                fg_resp = fg_match - baseline

                binary_fg_mask = (fg_resp > mmodel_thrd).type(torch.FloatTensor)

            binary_fg_mask = torch.squeeze(binary_fg_mask)

            if bmask_post_process:
                self.binary_mask_post_process(binary_fg_mask, cntxt_pad=cntxt_pad)

            return x.cpu().detach().numpy().squeeze(), binary_fg_mask.cpu().detach().numpy()

    def get_context_response(self, x, category):

        with torch.no_grad():

            context_resp = self.context_conv1o1(x)
            context_resp = context_resp[:, category * context_cluster: (category + 1) * context_cluster, :, :]
            context_resp, _ = context_resp.max(1)

        return context_resp


    def get_best_mixture(self, x, clutter_match, category):    # original


        if self.train:
            fg_model = self.center_crop(self.fg_models[category], x.shape[2:])
            fg_prior = self.center_crop(self.fg_prior[category], x.shape[2:])
            context_model = self.center_crop(self.context_models[category], x.shape[2:])
            context_prior = self.center_crop(self.context_prior[category], x.shape[2:])

            fused_model = (1 - self.omega) * fg_model * fg_prior.unsqueeze(1) + self.omega * context_model * context_prior.unsqueeze(1)
        else:
            fused_model = self.center_crop(self.fused_models[category], x.shape[2:])

        fg_match = self.log((x * fused_model).sum(1))

        occ = (clutter_match - fg_match > 0).type(torch.FloatTensor).cuda(device_ids[0])
        not_occ = 1.0 - occ

        view_point_scores = (not_occ * (fg_match - clutter_match)).sum((1, 2)) / (x.shape[2] * x.shape[3])
        k_max = int(torch.argmax(view_point_scores).item())

        return k_max, view_point_scores.max()

    def get_best_mixture_need_to_change_back(self, x, clutter_match, category):    # modified - 7/4/2021


        fg_model = self.center_crop(self.fg_models[category], x.shape[2:])
        fg_prior = self.center_crop(self.fg_prior[category], x.shape[2:])
        context_model = self.center_crop(self.context_models[category], x.shape[2:])
        context_prior = self.center_crop(self.context_prior[category], x.shape[2:])

        fg_model_with_prior = fg_model * fg_prior.unsqueeze(1)

        context_model_with_prior = context_model * context_prior.unsqueeze(1)

        fg_match = self.log((x * fg_model_with_prior).sum(1))               #(M, H, W)
        context_match = self.log((x * context_model_with_prior).sum(1))     #(M, H, W)

        final_match, _ = torch.stack([fg_match, context_match, clutter_match.repeat(fg_match.shape[0], 1, 1)]).max(0)
        view_point_scores = (final_match - clutter_match).sum((1, 2)) / (x.shape[2] * x.shape[3])

        k_max = int(torch.argmax(view_point_scores).item())

        return k_max, view_point_scores.max()

    def binary_mask_post_process(self, bmask, obj_prior='rigid', cntxt_pad=0):

        h, w = bmask.shape

        if cntxt_pad > 0:
            bmask[0:cntxt_pad, :] = 0
            bmask[:, 0:cntxt_pad] = 0
            bmask[h - cntxt_pad:h, :] = 0
            bmask[:, w - cntxt_pad:w] = 0

        if obj_prior == 'rigid':
            for i in range(h):
                for j in range(w):

                    if bmask[i][j] == 1:
                        continue

                    if torch.any(bmask[0:i, j] == 1) and torch.any(bmask[i + 1:, j] == 1) and torch.any(
                            bmask[i, 0:j] == 1) and torch.any(bmask[i, j + 1:] == 1):
                        bmask[i][j] = 1




    #====================== Helper / Setting Methods ======================#

    def center_crop(self, model, dim):

        axis_num = len(model.shape)

        if axis_num < 2 or axis_num > 4:
            print('center crop operation is only supported for axis_num = [2, 4].')
            raise

        h_crop, w_crop = dim
        h, w = model.shape[axis_num - 2 :]

        if h_crop > h or w_crop > w:
            diff = int((max(h_crop - h, w_crop - w) + 1) / 2)
            pad_ = [0 for i in range(axis_num * 2)]
            pad_[0:4] = [diff, diff, diff, diff]
            model = F.pad(model, tuple(pad_), 'constant', 0)
            h, w = model.shape[axis_num - 2 :]

        assert h_crop <= h and w_crop <= w

        if axis_num == 2:
            return model[int((h - h_crop) / 2): int((h + h_crop) / 2), int((w - w_crop) / 2): int((w + w_crop) / 2)]

        if axis_num == 3:
            return model[:, int((h - h_crop) / 2): int((h + h_crop) / 2), int((w - w_crop) / 2): int((w + w_crop) / 2)]

        if axis_num == 4:
            return model[:, :, int((h - h_crop) / 2): int((h + h_crop) / 2), int((w - w_crop) / 2): int((w + w_crop) / 2)]

    def update_mixture_models(self, Mixture_Models):

        fg_models, fg_priors, context_models, context_priors = Mixture_Models

        self.pig_fg_model = fg_models
        self.pig_fg_prior = fg_priors
        self.pig_context_model = context_models
        self.pig_context_prior = context_priors

        self.fg_models = [self.pig_fg_model]
        self.fg_prior = [self.pig_fg_prior]
        self.context_models = [self.pig_context_model]

        self.context_prior = [self.pig_context_prior]

    def remove_individual_models(self):
        self.pig_fg_model = None
        self.pig_fg_prior= None
        self.pig_context_model = None
        self.aeroplane_context_prior = None

    def update_fused_models(self, omega=0, type='ours'):

        self.fused_models = []

        for category in range(len(self.fg_models)):
            fg_model = self.fg_models[category]
            if fg_model == None:
                self.fused_models.append(None)
                continue
            fg_prior = self.fg_prior[category]
            context_model = self.context_models[category]
            context_prior = self.context_prior[category]
            if type == 'ours':
                self.fused_models.append((1 - omega) * fg_model * fg_prior.unsqueeze(1) + omega * context_model * context_prior.unsqueeze(1))
            elif type == 'standard':
                self.fused_models.append(fg_model * fg_prior.unsqueeze(1) + context_model * context_prior.unsqueeze(1))
            elif type == 'CA':
                self.fused_models.append((1 - omega) * fg_model + omega * context_model)


    def log(self, x, baseline=1e-6):

        return torch.log(self.relu(x) + baseline)

    def train(self):
        self.train = True

    def eval(self):
        self.train = False




    #====================== Additional Classes ======================#

class Conv1o1Layer(nn.Module):
    def __init__(self, weights):
        super(Conv1o1Layer, self).__init__()
        self.weight = nn.Parameter(weights)

    def forward(self, x, norm=True):
        if norm:
            norm = torch.norm(x, dim=1, keepdim=True)
            x = x / norm
        return F.conv2d(x, self.weight)

class ExpLayer(nn.Module):
    def __init__(self, vMF_kappa):
        super(ExpLayer, self).__init__()
        self.vMF_kappa = nn.Parameter(torch.Tensor([vMF_kappa]))

    def forward(self, x, binary=False):
        if binary:
            x = torch.exp(self.vMF_kappa * x) * (x > 0.55).type(torch.FloatTensor).cuda(device_ids[0])
        else:
            x = torch.exp(self.vMF_kappa * x)
        return x

class SoftMax(nn.Module):
    def __init__(self, constant):
        super(SoftMax, self).__init__()
        self.c = constant

    def forward(self, x):
        x = torch.exp(torch.clamp(x*self.c, -88.7, 88.7))
        return x / torch.sum(x, axis=1, keepdim=True)

class MedianPool2d(nn.Module):


    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x




