import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lap>=0.5.12")  # https://github.com/gatagat/lap
    import lap

def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """Calculate the intersection over box2 area given box1 and box2.

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes in x1y1x2y2 format.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes in x1y1x2y2 format.
        iou (bool, optional): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A numpy array of shape (N, M) representing the intersection over box2 area.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
            np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
    """Perform linear assignment using either the scipy or lap.lapjv method.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool): Use lap.lapjv for the assignment. If False, scipy.optimize.linear_sum_assignment is used.

    Returns:
        matched_indices (list[list[int]] | np.ndarray): Matched indices of shape (K, 2), where K is the number of
            matches.
        unmatched_a (np.ndarray): Unmatched indices from the first set, with shape (L,).
        unmatched_b (np.ndarray): Unmatched indices from the second set, with shape (M,).

    Examples:
        >>> cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> thresh = 5.0
        >>> matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh, use_lap=True)
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # Use lap.lapjv
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # Use scipy.optimize.linear_sum_assignment
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


# def iou_distance_ORI(atracks: list, btracks: list) -> np.ndarray:
#     """Compute cost based on Intersection over Union (IoU) between tracks.
#
#     Args:
#         atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
#         btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.
#
#     Returns:
#         (np.ndarray): Cost matrix computed based on IoU with shape (len(atracks), len(btracks)).
#
#     Examples:
#         Compute IoU distance between two sets of tracks
#         >>> atracks = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
#         >>> btracks = [np.array([5, 5, 15, 15]), np.array([25, 25, 35, 35])]
#         >>> cost_matrix = iou_distance(atracks, btracks)
#     """
#     if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
#         atlbrs = atracks
#         btlbrs = btracks
#     else:
#         atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
#         btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]
#
#     ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
#     if len(atlbrs) and len(btlbrs):
#         if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
#             ious = batch_probiou(
#                 np.ascontiguousarray(atlbrs, dtype=np.float32),
#                 np.ascontiguousarray(btlbrs, dtype=np.float32),
#             ).numpy()
#         else:
#             ious = bbox_ioa(
#                 np.ascontiguousarray(atlbrs, dtype=np.float32),
#                 np.ascontiguousarray(btlbrs, dtype=np.float32),
#                 iou=True,
#             )
#     return 1 - ious  # cost matrix


def iou_distance(atracks: list, btracks: list, metric: str = "iou") -> np.ndarray:
    """Compute cost based on IoU/GIoU/DIoU/CIoU between tracks.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes (x1,y1,x2,y2) or rotated (5).
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes (x1,y1,x2,y2) or rotated (5).
        metric (str): One of 'iou', 'giou', 'diou', 'ciou', 'hiou'. Defaults to 'iou'.

    Returns:
        (np.ndarray): Cost matrix (1 - similarity) with shape (len(atracks), len(btracks)).
    """
    eps = 1e-7
    # Extract box representations
    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    n, m = len(atlbrs), len(btlbrs)
    sims = np.zeros((n, m), dtype=np.float32)
    if n == 0 or m == 0:
        return 1 - sims

    # If boxes are 5-length (rotated/probabilistic), fallback to batch_probiou for IoU only
    if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
        ious = batch_probiou(
            np.ascontiguousarray(atlbrs, dtype=np.float32),
            np.ascontiguousarray(btlbrs, dtype=np.float32),
        ).numpy()
        if metric != "iou":
            # GIoU/DIoU/CIoU not implemented for rotated/probabilistic boxes here; fallback to IoU
            pass
        sims = ious
        return 1 - sims

    # Convert lists to arrays (N,4) and (M,4)
    a = np.ascontiguousarray(atlbrs, dtype=np.float32).reshape(n, 4)
    b = np.ascontiguousarray(btlbrs, dtype=np.float32).reshape(m, 4)
    a_x1, a_y1, a_x2, a_y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    b_x1, b_y1, b_x2, b_y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    # Intersection
    inter_w = (np.minimum(a_x2[:, None], b_x2) - np.maximum(a_x1[:, None], b_x1)).clip(0)
    inter_h = (np.minimum(a_y2[:, None], b_y2) - np.maximum(a_y1[:, None], b_y1)).clip(0)
    inter = inter_w * inter_h

    # Areas and union
    area_a = ((a_x2 - a_x1) * (a_y2 - a_y1))[:, None]
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    union = area_a + area_b - inter
    iou = inter / (union + eps)

    if metric == "iou":
        sims = iou
    elif metric == "hiou":
        a_h = (a_y2 - a_y1)[:, None]
        b_h = (b_y2 - b_y1)[None, :]
        union_h = a_h + b_h - inter_h
        sims = inter_h / (union_h + eps)
    else:
        # Enclosing box for GIoU/DIoU/CIoU
        enclose_x1 = np.minimum(a_x1[:, None], b_x1)
        enclose_y1 = np.minimum(a_y1[:, None], b_y1)
        enclose_x2 = np.maximum(a_x2[:, None], b_x2)
        enclose_y2 = np.maximum(a_y2[:, None], b_y2)
        enclose_w = (enclose_x2 - enclose_x1).clip(0)
        enclose_h = (enclose_y2 - enclose_y1).clip(0)
        enclose_area = enclose_w * enclose_h + eps

        if metric == "giou":
            sims = iou - (enclose_area - union) / enclose_area
        else:
            # DIoU and CIoU need center distances and diagonal of enclosing box
            a_cx = ((a_x1 + a_x2) * 0.5)[:, None]
            a_cy = ((a_y1 + a_y2) * 0.5)[:, None]
            b_cx = ((b_x1 + b_x2) * 0.5)[None, :]
            b_cy = ((b_y1 + b_y2) * 0.5)[None, :]
            center_dist2 = (a_cx - b_cx) ** 2 + (a_cy - b_cy) ** 2

            enclose_diag2 = enclose_w ** 2 + enclose_h ** 2 + eps
            diou = iou - center_dist2 / enclose_diag2

            if metric == "diou":
                sims = diou
            elif metric == "ciou":
                # aspect ratio term
                a_w = (a_x2 - a_x1)[:, None] + eps
                a_h = (a_y2 - a_y1)[:, None] + eps
                b_w = (b_x2 - b_x1)[None, :] + eps
                b_h = (b_y2 - b_y1)[None, :] + eps

                v = (4 / (np.pi ** 2)) * (np.arctan(b_w / b_h) - np.arctan(a_w / a_h)) ** 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = v / (1 - iou + v + eps)
                ciou = diou - alpha * v
                sims = ciou
            else:
                # unknown metric: fallback to iou
                sims = iou

    # clip to valid range just in case and return cost
    sims = np.clip(sims, -1.0, 1.0)
    return 1 - sims


def embedding_distance(tracks: list, detections: list, metric: str = "euclidean") -> np.ndarray:
    """Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list[STrack]): List of tracks, where each track contains embedding features.
        detections (list[BaseTrack]): List of detections, where each detection contains embedding features.
        metric (str): Metric for distance computation. Supported metrics include 'cosine', 'euclidean', etc.

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings with shape (N, M), where N is the number of tracks and M
            is the number of detections.

    Examples:
        Compute the embedding distance between tracks and detections using cosine metric
        >>> tracks = [STrack(...), STrack(...)]  # List of track objects with embedding features
        >>> detections = [BaseTrack(...), BaseTrack(...)]  # List of detection objects with embedding features
        >>> cost_matrix = embedding_distance(tracks, detections, metric="cosine")
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    for i, track in enumerate(tracks):
        cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))

        # DEBUG:
        # print("*" * 50)
        # print(track.smooth_feat.reshape(1,-1))
        # print("*" * 50)
        # print(det_features)
        # print("*" * 50)
        # print(det_features)
        # print("*" * 50)

    # track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    # cost_matrix    = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """Fuse cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        detections (list[BaseTrack]): List of detections, each containing a score attribute.

    Returns:
        (np.ndarray): Fused similarity matrix with shape (N, M).

    Examples:
        Fuse a cost matrix with detection scores
        >>> cost_matrix = np.random.rand(5, 10)  # 5 tracks and 10 detections
        >>> detections = [BaseTrack(score=np.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = det_scores[None].repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
