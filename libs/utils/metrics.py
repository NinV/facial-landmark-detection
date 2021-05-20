import numpy as np


def normalized_mean_error(gts, preds, d_ids):
    """
    gt, pred: [{kp1_id: [x1, y1],... }] or [[[x1, y1], [x2, y2], ...]]
    d_ids: landmark indexes ([kpId1, kpId2]) to calculate normalization distance d. The normalization distance i is
    the distance between 2 landmarks selected from kpId in ground truth
    """
    errors = {}

    for gt_i, pred_i in zip(gts, preds):
        d = np.linalg.norm(np.array(gt_i[d_ids[0]]) - np.array(gt_i[d_ids[1]]))
        if isinstance(gt_i, dict):
            for kpId in gt_i.keys():
                try:
                    errors[kpId].append(np.linalg.norm(np.array(gt_i[kpId]) - np.array(pred_i[kpId])) / d)
                except KeyError:
                    errors[kpId] = [np.linalg.norm(np.array(gt_i[kpId]) - np.array(pred_i[kpId])) / d]
        else:
            for l, (gt_i_l, pred_i_l) in enumerate(zip(gt_i, pred_i)):
                try:
                    errors[l].append(np.linalg.norm(np.array(gt_i_l) - np.array(pred_i_l)) / d)
                except KeyError:
                    errors[l] = [np.linalg.norm(np.array(gt_i_l) - np.array(pred_i_l)) / d]
    mean_errors_per_landmark = {kpId: np.mean(err) for kpId, err in errors.items()}
    nme = np.mean(list(mean_errors_per_landmark.values()))
    return nme, errors, mean_errors_per_landmark


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
