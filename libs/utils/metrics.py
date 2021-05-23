import numpy as np


def root_mean_square_error(gts, preds):
    """
        gt, pred: [{kp1_id: [x1, y1, c],... }]
    """
    errors = {}
    for gt_i, pred_i in zip(gts, preds):
        for kpId in gt_i.keys():
            try:
                errors[kpId].append(np.linalg.norm(np.array(gt_i[kpId]) - np.array(pred_i[kpId])))
            except KeyError:
                errors[kpId] = [np.linalg.norm(np.array(gt_i[kpId]) - np.array(pred_i[kpId]))]

    mean_errors_per_landmark = {kpId: np.mean(err) for kpId, err in errors.items()}
    rmse = np.mean(list(mean_errors_per_landmark.values()))
    return rmse, errors, mean_errors_per_landmark


def compute_nme(preds, meta):
    """
    source: https://github.com/HRNet/HRNet-Facial-Landmark-Detection
    """
    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse
