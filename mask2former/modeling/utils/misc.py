import torch
import torch.nn.functional as F

def mask_avg_pool(feature, mask, use_gt=False, return_mask=False):
    '''
    Input: 
        feature: (Bs, C, H1, W1)
        msak: (Bs, P, H2, W2)
    Output:
        (Bs, P, C), 1, 1)
    '''
    Bs, feat_c, feat_h, feat_w = feature.shape
    Bs, mask_p, mask_h, mask_w = mask.shape

    # import pdb; pdb.set_trace()
    # if feat_h != mask_h or feat_w != mask_w:
    if use_gt:
        mask = F.interpolate(mask.float(), size=(feat_h, feat_w), mode='nearest')
    else:
        mask = F.interpolate(mask, size=(feat_h, feat_w), mode='bilinear')
        mask = torch.sigmoid(mask).detach()
        mask = torch.where(mask > 0.3, 1., 0.)

    if return_mask:
        valid_mask = mask.sum(dim=(-1, -2))
        valid_mask = torch.where(valid_mask > 0.5, 1, 0)

    # import pdb; pdb.set_trace()

    feature = feature[:, None, :, :, :].repeat(1, mask_p, 1, 1, 1) # (Bs, P, C, h1, w1)
    mask = mask[:, :, None, :, :].repeat(1, 1, feat_c, 1, 1) # (Bs, P, C, h1, w1)

    mask_feat = F.adaptive_avg_pool2d(feature * mask, (1, 1)) # (Bs, P, C)
    mask_feat = mask_feat.squeeze(-2).squeeze(-1)

    mask_weight = feat_h * feat_w / (torch.sum(mask, (-2, -1)) + 1e-6) # (Bs, P, C)

    # import pdb;pdb.set_trace()

    if return_mask:
        return mask_feat * mask_weight, valid_mask
    else:
        return mask_feat * mask_weight # (Bs, P, C)


def get_sup_propotype(feature, mask, use_gt=False, return_mask=False):
    '''
    Input: 
        feature: (Bs, C, H1, W1)
        msak: (Bs, P, H2, W2)
    Output:
        (Bs, P, C), 1, 1)
    '''
    Bs, feat_c, feat_h, feat_w = feature.shape
    Bs, mask_p, mask_h, mask_w = mask.shape

    # import pdb; pdb.set_trace()
    # if feat_h != mask_h or feat_w != mask_w:
    if use_gt:
        mask = F.interpolate(mask.float(), size=(feat_h, feat_w), mode='nearest')
    else:
        mask = F.interpolate(mask, size=(feat_h, feat_w), mode='bilinear')
        mask = torch.sigmoid(mask).detach()
        mask = torch.where(mask > 0.3, 1., 0.)

    if return_mask:
        valid_mask = mask.sum(dim=(-1, -2))
        valid_mask = torch.where(valid_mask > 0.5, 1, 0)

    # import pdb; pdb.set_trace()

    feature = feature[:, None, :, :, :].repeat(1, mask_p, 1, 1, 1) # (Bs, P, C, h1, w1)
    mask = mask[:, :, None, :, :].repeat(1, 1, feat_c, 1, 1) # (Bs, P, C, h1, w1)

    weight = torch.sum(mask, dim=(3, 4)) # Bs, P, C

    mask_feat = torch.sum(feature * mask, dim=(3, 4)) + 1e-6 # (Bs, P, C)
    mask_feat = mask_feat / weight
    # mask_feat = mask_feat.squeeze(-2).squeeze(-1)

    # mask_weight = feat_h * feat_w / (torch.sum(mask, (-2, -1)) + 1e-6) # (Bs, P, C)

    # import pdb;pdb.set_trace()

    if return_mask:
        return mask_feat, valid_mask
    else:
        return mask_feat # (Bs, P, C)

    label = F.interpolate(label.unsqueeze(1), size=features_for_propotype.shape[-2:], mode='bilinear', align_corners=True)
    weight = torch.sum(label, dim = (2,3))  # bs* 1

    propotype = features_for_propotype * label  # bs* 256*h*w
    propotype = torch.sum(propotype, dim = (2,3))/ weight

    sup_propotype = propotype.unsqueeze(-1)

    return sup_propotype


def process_coco_cat(cls_name):
    if isinstance(cls_name, str):
        cls_name = [cls_name]
        
    cls_name = [n.replace("-merged", "") for n in cls_name]
    cls_name = [n.replace("-other", "") for n in cls_name]
    cls_name = [n.replace("-", " ") for n in cls_name]

    return cls_name