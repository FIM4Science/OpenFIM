import numpy as np
import torch


def find_alignment_mc(seq1, seq2, del_cost, trans_cost):
    """
    We use dynamic programming to find the best alignments between two seqs.
    ``nc'' means that this functions support a series of del_cost values.
    Note: Not support multiple types.
    :param np.ndarray seq1: Time stamps of seq #1.
    :param np.ndarray seq2: Time stamps of seq #2.
    :param np.ndarray del_cost: A series of delete cost.
    :param float trans_cost: Transportation cost per unit length.
    :return: Alignment list and minimum distances for all the del_cost values.
    """
    n_cost = len(del_cost)
    n1 = len(seq1)
    n2 = len(seq2)
    # shape=[n2, n1]
    trans_mask = np.abs(seq2.repeat(n1).reshape(n2, n1) - seq1) * trans_cost
    # shape=[n1+1, n1+1]
    del_mask = (
        np.arange(n1 + 2, dtype=np.float32).repeat(n1 + 1).reshape(n1 + 2, n1 + 1).T.reshape(-1)[: (n1 + 1) ** 2].reshape(n1 + 1, n1 + 1)
        - 1
    )
    del_mask[np.tril_indices(n1 + 1, -1)] = float("inf")
    # shape=[n1+1, n1+1, n_cost]
    del_mask = del_mask.repeat(n_cost).reshape(n1 + 1, n1 + 1, n_cost) * del_cost
    # shape=[n1+1, n1+1, n_cost]
    del_mask = del_mask.transpose([1, 0, 2]).copy()
    # shape=[n1+1, n_cost]
    overhead = np.empty(shape=[n1 + 1, n_cost], dtype=np.float32)
    overhead.fill(float("inf"))
    overhead[0, :] = 0.0
    # shape=[n2, n1+1, n_cost]
    back_pointers = np.empty(shape=[n2, n1 + 1, n_cost], dtype=np.int32)
    for n2_idx in range(n2):
        # shape=[n1+1, n1+1, n_cost]
        add_mask = del_mask.copy()
        add_mask[1:, :, :] += np.outer(trans_mask[n2_idx], np.ones(shape=[(n1 + 1) * n_cost], dtype=np.float32)).reshape(n1, n1 + 1, n_cost)
        add_mask[np.arange(n1 + 1), np.arange(n1 + 1), :] = del_cost
        # shape=[n1+1, n1+1, n_cost]
        cost_mat = overhead + add_mask
        # shape=[n1+1, n_cost]
        choices = np.argmin(cost_mat, axis=1)
        back_pointers[n2_idx] = choices
        overhead = cost_mat.min(axis=1)
    overhead += np.outer(np.arange(n1, -1, -1, dtype=np.float32), np.ones(shape=[n_cost])) * del_cost
    # shape=[n_cost]
    curr_choice = np.argmin(overhead, axis=0)
    # shape=[n_cost]
    min_distance = overhead.min(axis=0)
    best_route = [curr_choice]
    # shape=[n1+1, n_cost]
    for choice_list in back_pointers[::-1]:
        # shape=[n_cost]
        curr_choice = choice_list[curr_choice, np.arange(n_cost)]
        best_route.append(curr_choice)
    # shape=[n2, n_cost]
    best_route = np.array(best_route)

    align_pairs = []
    for cost_idx in range(n_cost):
        best_route_ = best_route[:, cost_idx]
        pairs = []
        memo = -1
        for n2_idx_plus_1, choice_made in enumerate(best_route_[::-1]):
            if choice_made != memo:
                pairs.append([choice_made - 1, n2_idx_plus_1 - 1])
            memo = choice_made
        align_pairs.append(pairs[1:])

    return [align_pairs, min_distance]


def distance_between_event_seq(ref_seq, decode_seq, del_cost, trans_cost, num_types):
    """
    Args:
        ref_seq: [time_seqs, event_seqs]
        decode_seq: [time_seqs, event_seqs]
        del_cost:
        trans_cost:
        num_types:

    Returns:

    """
    num_cost = len(del_cost)
    distances = np.zeros(shape=[num_cost], dtype=np.float32)
    total_trans_cost = np.zeros(shape=[num_cost], dtype=np.float32)
    num_true = np.zeros(shape=[num_cost], dtype=np.int32)
    num_del = np.zeros(shape=[num_cost], dtype=np.int32)
    num_ins = np.zeros(shape=[num_cost], dtype=np.int32)
    num_align = np.zeros(shape=[num_cost], dtype=np.int32)

    seq_per_types = [[[], []] for _ in range(num_types)]
    for seq_idx, seq in enumerate([ref_seq, decode_seq]):
        for event_time, event_type in zip(*seq):
            if event_type >= num_types:
                continue
            seq_per_types[int(event_type)][seq_idx].append(event_time)

    for type_idx in range(num_types):
        ref_time = np.array(seq_per_types[type_idx][0])
        decoded_time = np.array(seq_per_types[type_idx][1])
        align_pairs, min_distance = find_alignment_mc(ref_time, decoded_time, del_cost, trans_cost)
        for cost_idx in range(num_cost):
            align_pairs_per_cost = align_pairs[cost_idx]
            min_distance_per_cost = min_distance[cost_idx]
            num_align[cost_idx] += len(align_pairs_per_cost)
            num_true[cost_idx] += len(ref_time)
            n_ins_per_cost = len(decoded_time) - len(align_pairs_per_cost)
            n_del_per_cost = len(ref_time) - len(align_pairs_per_cost)
            num_ins[cost_idx] += n_ins_per_cost
            num_del[cost_idx] += n_del_per_cost
            distances[cost_idx] += min_distance_per_cost
            total_trans_cost[cost_idx] += min_distance_per_cost - del_cost[cost_idx] * (n_ins_per_cost + n_del_per_cost)
    return distances, total_trans_cost, num_true, num_del, num_ins, num_align


def get_distances_otd(pred_dt, pred_type_result, gt_dt, gt_type_result, num_classes, distance_del_cost, trans_cost):
    """
    Wrapper for Optimal Transport Distance calculation.
    This version expects INTER-ARRIVAL times (dtimes) and converts them to
    absolute timestamps internally before calculating the distance.
    """

    distances = []

    for pred_time, pred_type, gt_time, gt_type in zip(pred_dt, pred_type_result, gt_dt, gt_type_result):
        # Convert inter-arrival times to absolute timestamps
        pred_abs_time = torch.cumsum(pred_time, dim=-1)
        gt_abs_time = torch.cumsum(gt_time, dim=-1)

        ref_seq = [pred_abs_time.cpu().numpy(), pred_type.cpu().numpy()]
        decode_seq = [gt_abs_time.cpu().numpy(), gt_type.cpu().numpy()]

        distance = distance_between_event_seq(ref_seq, decode_seq, distance_del_cost, trans_cost, num_classes)[0]
        distances.append(distance)

    return distances
