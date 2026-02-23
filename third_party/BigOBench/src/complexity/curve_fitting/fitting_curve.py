# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .fitting_tree import ComplexityCandidate, NodeComplexity, GroupComplexity
import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import statistics
import sys


# ============================
# NEW: robust value extraction
# ============================
def _get_values_for_aggregation(value_details: dict):
    """
    Prefer robust aggregated value (value_med) if available (worker-side warmup drop + median).
    Fallback to raw value_list for backward compatibility.

    Returns:
        list[float]: values usable by min/max/mean/median/first aggregation.
    """
    # New path: worker-added robust stats
    if isinstance(value_details, dict) and value_details.get("value_ok") and value_details.get("value_med") is not None:
        try:
            v = float(value_details["value_med"])
            if np.isfinite(v):
                return [v]
        except Exception:
            pass

    # Fallback path: old raw list
    vlist = []
    if isinstance(value_details, dict):
        vlist = value_details.get("value_list", []) or []

    out = []
    for x in vlist:
        if x is None:
            continue
        try:
            xx = float(x)
            if np.isfinite(xx):
                out.append(xx)
        except Exception:
            continue
    return out


def is_there_peak(x_list, y_list):
    """
    Checks if there is a peak in the given data.
    A peak is defined as a point where the y-value is greater than the next y-value,
    but only considers points where the corresponding x-value is 256 or greater.
    """
    found_peak = False
    for i in range(len(y_list) - 1):
        if x_list[i] < 256:
            continue
        if y_list[i] > y_list[i + 1]:
            found_peak = True
    return found_peak


def min_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    Aggregate by taking the minimum value per x.
    Supports worker robust fields: value_med/value_ok.
    """
    x_list, y_list = [], []
    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        vals = _get_values_for_aggregation(value_details)
        if len(vals) == 0:
            continue
        y_list.append(min(vals) * (10000 if enlarge_values else 1))
        x_list.append(int(multiplier_value))
    return x_list, y_list


def max_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    Aggregate by taking the maximum value per x.
    Supports worker robust fields: value_med/value_ok.
    """
    x_list, y_list = [], []
    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        vals = _get_values_for_aggregation(value_details)
        if len(vals) == 0:
            continue
        y_list.append(max(vals) * (10000 if enlarge_values else 1))
        x_list.append(int(multiplier_value))
    return x_list, y_list


def mean_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    Aggregate by taking the mean value per x.
    Supports worker robust fields: value_med/value_ok.
    """
    x_list, y_list = [], []
    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        vals = _get_values_for_aggregation(value_details)
        if len(vals) == 0:
            continue
        y_list.append(float(np.mean(vals)) * (10000 if enlarge_values else 1))
        x_list.append(int(multiplier_value))
    return x_list, y_list


def median_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    Aggregate by taking the median value per x.
    Supports worker robust fields: value_med/value_ok.
    """
    x_list, y_list = [], []
    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        vals = _get_values_for_aggregation(value_details)
        if len(vals) == 0:
            continue
        y_list.append(float(statistics.median(vals)) * (10000 if enlarge_values else 1))
        x_list.append(int(multiplier_value))
    return x_list, y_list


def first_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    Aggregate by taking the first recorded value per x.
    Supports worker robust fields: value_med/value_ok (treated as the only value).
    """
    x_list, y_list = [], []
    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        vals = _get_values_for_aggregation(value_details)
        if len(vals) == 0:
            continue
        y_list.append(vals[0] * (10000 if enlarge_values else 1))
        x_list.append(int(multiplier_value))
    return x_list, y_list


def most_stable_run_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    Original intent: choose the most stable single-run curve (per-run index).
    With worker robust aggregation (value_med), stability is already handled upstream.

    So:
      - if any point has value_med -> return median_aggregate
      - else fallback to original per-run logic
    """
    any_has_med = any(
        (isinstance(v, dict) and v.get("value_med") is not None)
        for v in multiplier_value_to_value_details_dict.values()
    )
    if any_has_med:
        return median_aggregate(multiplier_value_to_value_details_dict, enlarge_values)

    # ---------- OLD behavior ----------
    first_vals = None
    for v in multiplier_value_to_value_details_dict.values():
        if not isinstance(v, dict):
            continue
        if "value_list" in v and isinstance(v["value_list"], list) and len(v["value_list"]) > 0:
            first_vals = v["value_list"]
            break

    if first_vals is None:
        return [], []

    number_runs = len(first_vals)
    x_list_list = [[] for _ in range(number_runs)]
    y_list_list = [[] for _ in range(number_runs)]

    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        if not isinstance(value_details, dict):
            continue
        vlist = value_details.get("value_list", []) or []
        if len(vlist) == 0:
            continue

        for i, value in enumerate(vlist):
            if i >= number_runs:
                continue
            if value is None:
                continue
            try:
                vv = float(value)
            except Exception:
                continue

            x_list_list[i].append(int(multiplier_value))
            y_list_list[i].append(vv * (10000 if enlarge_values else 1))

    # we default to the median if that does not work
    for x_list, y_list in zip(x_list_list, y_list_list):
        if len(x_list) == 0:
            continue
        if not is_there_peak(x_list, y_list):
            return x_list, y_list

    return median_aggregate(multiplier_value_to_value_details_dict, enlarge_values)


def most_stable_aggregate_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    Try min, then max, then median, returning the first that doesn't show a peak.
    Uses updated min/max/median which support worker robust fields.
    """
    x_list, y_list = min_aggregate(multiplier_value_to_value_details_dict, enlarge_values)
    if not is_there_peak(x_list, y_list):
        return x_list, y_list

    x_list, y_list = max_aggregate(multiplier_value_to_value_details_dict, enlarge_values)
    if not is_there_peak(x_list, y_list):
        return x_list, y_list

    x_list, y_list = median_aggregate(multiplier_value_to_value_details_dict, enlarge_values)
    return x_list, y_list


def _safe_fit_complexity(
    complexity_instance,
    x_list_filtered,
    y_list_filtered,
    apply_constraints,
    zero_out_first_value,
    piecewise_fit,
    max_time_x_value,
    aggressive_max_time_x_scaling=True,
    residual_mode="relative",
    eps=1e-12,
):
    """
    Call ComplexityClass.fit with backward compatibility:
    - If fit() supports residual_mode/eps -> use it
    - Else fallback to old signature
    """
    try:
        return complexity_instance.fit(
            x_list_filtered,
            y_list_filtered,
            apply_constraints=apply_constraints,
            zero_out_first_value=zero_out_first_value,
            piecewise_fit=piecewise_fit,
            max_time_x_value=max_time_x_value,
            residual_mode=residual_mode,
            eps=eps,
        )
    except TypeError:
        # old fitting_class.py does not accept residual_mode/eps
        return complexity_instance.fit(
            x_list_filtered,
            y_list_filtered,
            apply_constraints=apply_constraints,
            zero_out_first_value=zero_out_first_value,
            piecewise_fit=piecewise_fit,
            max_time_x_value=max_time_x_value,
        )


def infer_complexity_from_values(
    value_dict,
    complexity_name_function_list,
    filter_outliers=True,
    apply_penalty=True,
    apply_constraints=True,
    zero_out_first_value=True,
    piecewise_fit=True,
    aggregate_y_values='min_aggregate',
    max_time_rate=0.8,
    elect_complexity='min',
    fix_constant_complexity=True,
    fix_negligeable_complexity=True,
    enlarge_values=True,
    print_info=True,
    multiplier_start=1,
    aggressive_max_time_x_scaling=True,
    # NEW knobs
    residual_mode="relative",        # "relative" recommended
    tie_ratio_threshold=1.05,        # top2 close threshold
    tie_grow_threshold=1.10,         # t/n growth threshold -> pick nlogn
    peak_penalty=1.2,                # penalty multiplier for peak curves
    min_points_gate=4,               # require at least this many x points
):
    """
    Infers complexity from measured runtime/memory curves.

    Returns:
        (complexity_str, found_peak_deterministic, coeff1, coeff2, any_peak)
    """

    # -----------------------
    # Aggregation selection
    # -----------------------
    assert isinstance(aggregate_y_values, str)
    assert aggregate_y_values in [
        'min_aggregate',
        'max_aggregate',
        'median_aggregate',
        'mean_aggregate',
        'first_aggregate',
        'most_stable_run_aggregate',
        'most_stable_aggregate_aggregate',
    ]

    if aggregate_y_values == 'min_aggregate':
        aggregate_function = min_aggregate
    elif aggregate_y_values == 'max_aggregate':
        aggregate_function = max_aggregate
    elif aggregate_y_values == 'mean_aggregate':
        aggregate_function = mean_aggregate
    elif aggregate_y_values == 'median_aggregate':
        aggregate_function = median_aggregate
    elif aggregate_y_values == 'first_aggregate':
        aggregate_function = first_aggregate
    elif aggregate_y_values == 'most_stable_run_aggregate':
        aggregate_function = most_stable_run_aggregate
    elif aggregate_y_values == 'most_stable_aggregate_aggregate':
        aggregate_function = most_stable_aggregate_aggregate
    else:
        raise Exception('not supported')

    assert elect_complexity in ['min', 'max']
    if elect_complexity == 'min':
        elect_function = np.argmin
    else:
        elect_function = np.argmax

    complexity_details_list = []

    # estimate global max x for scaling
    max_time_x_value = list(map(lambda x: max(x) if len(x) > 0 else 0, list(map(
        lambda multiplier_value_to_value_details_list_dict: [
            int(multiplier_value)
            for multiplier_value, value_details_list in multiplier_value_to_value_details_list_dict.items()
            if len([vd for vd in value_details_list if len(_get_values_for_aggregation(vd)) > 0]) > 0
        ],
        value_dict.values()
    ))))
    max_time_x_value = max(max_time_x_value) if len(max_time_x_value) > 0 else 0

    if print_info:
        print(max_time_x_value)

    found_peak_max_time_list = []

    # helpers for tie-break label detection
    def _is_linear(name: str) -> bool:
        s = str(name).lower()
        # allow multiple possible label styles
        if s in ["o(n)", "n", "linear"]:
            return True
        # "o(n)" but not containing log
        return ("o(n" in s and "log" not in s and "n^" not in s and "n²" not in s)

    def _is_nlogn(name: str) -> bool:
        s = str(name).lower()
        if s in ["o(nlogn)", "o(n log n)", "nlogn"]:
            return True
        return ("log" in s and "n" in s)

    for variable_index, (variable_name_type_dimension, multiplier_value_to_value_details_list_dict) in enumerate(value_dict.items()):
        id_set = set(map(lambda x: x['id_'], [
            value_details
            for value_details_list in list(multiplier_value_to_value_details_list_dict.values())
            for value_details in value_details_list
        ]))

        if print_info:
            print('####################################')
            print(variable_name_type_dimension, 'methods', id_set)

        id_set = sorted(list(id_set))

        for id_ in id_set:
            found_peak = False
            max_time_peak = 0

            if print_info:
                print('############', variable_name_type_dimension, id_)

            multiplier_value_to_value_details_dict = {}
            for multiplier_value, value_details_list in multiplier_value_to_value_details_list_dict.items():
                filtered_value_details_list = list(filter(lambda x: x['id_'] == id_, value_details_list))
                if len(filtered_value_details_list) == 1:
                    multiplier_value_to_value_details_dict[multiplier_value] = filtered_value_details_list[0]
                elif len(filtered_value_details_list) > 1:
                    if print_info:
                        print('weird, skipping...')
                        print(filtered_value_details_list)
                    raise Exception('')

            if len(list(multiplier_value_to_value_details_dict.keys())) <= 3:
                if print_info:
                    print('skipping a method', 1)
                continue

            residuals_list = []
            ref_t_list = []
            t_list = []
            max_time_list = []
            coeff_list = []

            tag_list = None
            priority = None

            # pick tag_list/priority from first usable point
            for _, value_details in multiplier_value_to_value_details_dict.items():
                vals = _get_values_for_aggregation(value_details)
                if len(vals) == 0:
                    continue
                tag_list = value_details.get('tag_list', None)
                priority = value_details.get('priority', None)
                break

            x_list, y_pulled_list = aggregate_function(multiplier_value_to_value_details_dict, enlarge_values)

            if len(x_list) <= 3:
                if print_info:
                    print('skipping a method', 2)
                continue

            if x_list[0] != multiplier_start:
                if print_info:
                    print('skipping a method', 1)
                continue

            # normalize first point
            y_pulled_list[0] = min(y_pulled_list)

            # detect peak (raw pulled curve)
            for i in range(len(y_pulled_list) - 1):
                if x_list[i] < 256:
                    continue
                if y_pulled_list[i] > y_pulled_list[i + 1]:
                    found_peak = True

            # filter outliers (original logic)
            if filter_outliers:
                x_list_filtered_tmp, y_list_filtered_tmp = [], []
                for i in range(0, len(y_pulled_list)):
                    if y_pulled_list[i] / y_pulled_list[-1] < 1.3:
                        x_list_filtered_tmp.append(x_list[i])
                        y_list_filtered_tmp.append(y_pulled_list[i])
                x_list = x_list_filtered_tmp[:]
                y_pulled_list = y_list_filtered_tmp[:]

            if len(x_list) <= 3:
                if print_info:
                    print('skipping a method', 3)
                continue

            # monotonic "offset" adjustment (original logic)
            if filter_outliers:
                x_list_filtered = []
                y_list_filtered = []
                x_start = 512
                x_list_filtered.append(x_list[0])
                y_list_filtered.append(y_pulled_list[0])
                offset = 0

                for i in range(1, len(y_pulled_list) - 1):
                    if x_list[i] <= x_start:
                        x_list_filtered.append(x_list[i])
                        y_list_filtered.append(y_pulled_list[i])
                    else:
                        if y_pulled_list[i - 1] <= y_pulled_list[i]:
                            x_list_filtered.append(x_list[i])
                            y_list_filtered.append(y_pulled_list[i] + offset)
                        else:
                            # temp peak: skip
                            if y_pulled_list[i - 1] < y_pulled_list[i + 1]:
                                continue
                            # permanent peak: stop
                            elif y_pulled_list[i] < y_pulled_list[i + 1]:
                                break
                            else:
                                x_list_filtered.append(x_list[i])
                                y_list_filtered.append(y_pulled_list[i] + offset)

                else:
                    if y_pulled_list[-2] <= y_pulled_list[-1]:
                        x_list_filtered.append(x_list[-1])
                        y_list_filtered.append(y_pulled_list[-1] + offset)
            else:
                x_list_filtered = x_list
                y_list_filtered = y_pulled_list

            if len(x_list_filtered) <= 2:
                if print_info:
                    print('skipping a method', 4)
                continue

            # =========================
            # Curve quality gate (NEW)
            # =========================
            if len(x_list_filtered) < min_points_gate:
                if print_info:
                    print("gate: too few points -> skip", id_, len(x_list_filtered))
                continue

            yy = np.asarray(y_list_filtered, dtype=float)
            if not np.all(np.isfinite(yy)):
                if print_info:
                    print("gate: non-finite y -> skip", id_)
                continue

            # almost-flat curve: likely noise
            y_span = float(np.max(yy) - np.min(yy))
            y_max = float(np.max(yy))
            if y_span <= max(1e-12, 0.01 * max(y_max, 1e-12)):
                if print_info:
                    print("gate: almost flat -> skip", id_, y_span, y_max)
                continue

            peak_penalty_factor = (peak_penalty if found_peak else 1.0)

            max_time_peak = y_list_filtered[-1]

            # -----------------------
            # Fit all candidate classes
            # -----------------------
            for complexity_name, complexity_class, penalty, order_ in complexity_name_function_list:
                complexity_instance = complexity_class()
                residuals, ref_t, t, max_time_temp, coeff = _safe_fit_complexity(
                    complexity_instance,
                    x_list_filtered,
                    y_list_filtered,
                    apply_constraints=apply_constraints,
                    zero_out_first_value=zero_out_first_value,
                    piecewise_fit=piecewise_fit,
                    max_time_x_value=(max_time_x_value if aggressive_max_time_x_scaling else min(max_time_x_value, max(x_list_filtered))),
                    aggressive_max_time_x_scaling=aggressive_max_time_x_scaling,
                    residual_mode=residual_mode,
                    eps=1e-12,
                )

                # sanity
                assert len(x_list_filtered) == len(ref_t)
                assert len(x_list_filtered) == len(t)

                max_time_list.append(max_time_temp)
                coeff_list.append(coeff)

                residuals_list.append(
                    (residuals + 0)
                    * (penalty if apply_penalty else 1)
                    * peak_penalty_factor
                )
                ref_t_list.append(ref_t)
                t_list.append(t)

            # constant class safeguard (original behavior)
            index_of_constant_complexity = list(map(lambda x: x[0], filter(
                lambda x: x[1] == 'o(1)',
                enumerate(map(lambda x: x[0], complexity_name_function_list))
            )))[0]

            if math.isclose(residuals_list[index_of_constant_complexity], 0, rel_tol=1e-8):
                residuals_list[index_of_constant_complexity] = 0
                for i in range(0, len(residuals_list)):
                    if i == index_of_constant_complexity:
                        continue
                    residuals_list[i] = max(residuals_list[i], 1e-5)

            # =========================
            # Tie-break (NEW): n vs nlogn
            # =========================
            sorted_idx = list(np.argsort(residuals_list))
            best_i = sorted_idx[0]
            second_i = sorted_idx[1] if len(sorted_idx) > 1 else None

            best_name = complexity_name_function_list[best_i][0]
            second_name = complexity_name_function_list[second_i][0] if second_i is not None else None

            if second_i is not None:
                r1 = float(residuals_list[best_i])
                r2 = float(residuals_list[second_i])
                if (r2 / max(r1, 1e-12)) < tie_ratio_threshold:
                    # only apply for the (linear, nlogn) pair
                    if (_is_linear(best_name) and _is_nlogn(second_name)) or (_is_nlogn(best_name) and _is_linear(second_name)):
                        xx = np.asarray(x_list_filtered, dtype=float)
                        yy = np.asarray(y_list_filtered, dtype=float)

                        ratio = yy / np.maximum(xx, 1.0)
                        if len(ratio) >= 4:
                            grow = float(np.median(ratio[-2:]) / max(np.median(ratio[:2]), 1e-12))
                            if print_info:
                                print("tie-break n vs nlogn: grow(t/n)=", grow, "best=", best_name, "second=", second_name)
                            if grow > tie_grow_threshold:
                                # choose nlogn
                                if _is_nlogn(best_name):
                                    pass
                                else:
                                    best_i = second_i
                                    best_name = complexity_name_function_list[best_i][0]

            # backup index (exclude constant)
            backup_i = [i for i in np.argsort(residuals_list) if complexity_name_function_list[i][0] != 'o(1)'][0]

            complexity_details_list.append(
                {
                    'variable_name_type_dimension': variable_name_type_dimension,
                    'class_': complexity_name_function_list[best_i][0],
                    'order': complexity_name_function_list[best_i][3],
                    'class_backup': complexity_name_function_list[backup_i][0],
                    'order_backup': complexity_name_function_list[backup_i][3],
                    'found_peak': found_peak,
                    'max_time': max_time_list[best_i],
                    'priority': priority,
                    'tag_list': tag_list,
                    'number_variables': len(tag_list) if tag_list is not None else 0,
                    'variable_index': variable_index,
                    'coeff': coeff_list[best_i],
                    'coeff_backup': coeff_list[backup_i],
                }
            )

            found_peak_max_time_list.append((found_peak, max_time_peak))

            if print_info:
                print('complexity', complexity_name_function_list[best_i][0])
                print('time', max_time_list[best_i])
                print('coeff', coeff_list[best_i])
                print('tag_list', tag_list)

            if print_info:
                print(residuals_list)
                p = plt.plot(x_list_filtered, ref_t_list[best_i], '--')
                p = plt.plot(
                    x_list_filtered,
                    t_list[best_i],
                    color=p[0].get_color(),
                    label=str(variable_index) + ', ' + id_
                )

    # -----------------------
    # Build tree & group
    # -----------------------
    complexity_details_list.sort(key=lambda x: (x['number_variables'], x['variable_index']))
    variable_name_type_dimension_to_complexity_group_dict = {}

    for variable_name_type_dimension in value_dict.keys():
        node = NodeComplexity(variable_name_type_dimension)
        node.print_info = print_info
        variable_name_type_dimension_to_complexity_group_dict[variable_name_type_dimension] = node

    if print_info:
        print(complexity_details_list)

    for complexity_details in complexity_details_list:
        if complexity_details['number_variables'] == 1:
            variable_name_type_dimension_to_complexity_group_dict[
                complexity_details['variable_name_type_dimension']
            ].add_complexity(ComplexityCandidate(
                class_=complexity_details['class_'],
                order=complexity_details['order'],
                class_backup=complexity_details['class_backup'],
                order_backup=complexity_details['order_backup'],
                max_time=complexity_details['max_time'],
                priority=complexity_details['priority'],
                found_peak=complexity_details['found_peak'],
                coeff=complexity_details['coeff'],
                coeff_backup=complexity_details['coeff_backup'],
            ))
        else:
            assert complexity_details['number_variables'] > 1
            complexity_group_list = [
                variable_name_type_dimension_to_complexity_group_dict[v]
                for v in complexity_details['tag_list']
            ]

            if len(set(complexity_group_list)) == 1:
                complexity_group_list[0].add_complexity(ComplexityCandidate(
                    class_=complexity_details['class_'],
                    order=complexity_details['order'],
                    class_backup=complexity_details['class_backup'],
                    order_backup=complexity_details['order_backup'],
                    max_time=complexity_details['max_time'],
                    priority=complexity_details['priority'],
                    found_peak=complexity_details['found_peak'],
                    coeff=complexity_details['coeff'],
                    coeff_backup=complexity_details['coeff_backup'],
                ))
            else:
                group_complexity = GroupComplexity()
                group_complexity.print_info = print_info

                for group_or_node in set(complexity_group_list):
                    group_complexity.add_group_or_node(group_or_node)

                group_complexity.add_complexity(ComplexityCandidate(
                    class_=complexity_details['class_'],
                    order=complexity_details['order'],
                    class_backup=complexity_details['class_backup'],
                    order_backup=complexity_details['order_backup'],
                    max_time=complexity_details['max_time'],
                    priority=complexity_details['priority'],
                    found_peak=complexity_details['found_peak'],
                    coeff=complexity_details['coeff'],
                    coeff_backup=complexity_details['coeff_backup'],
                ))

                for v in complexity_details['tag_list']:
                    variable_name_type_dimension_to_complexity_group_dict[v] = group_complexity

                for v, complexity_group in variable_name_type_dimension_to_complexity_group_dict.items():
                    if complexity_group in set(complexity_group_list):
                        variable_name_type_dimension_to_complexity_group_dict[v] = group_complexity

    # At least one variable must have some complexity candidates
    if not any([
        len(complexity_group.complexity_candidate_list) > 0
        for complexity_group in variable_name_type_dimension_to_complexity_group_dict.values()
    ]):
        return None, False, None, None, None

    if len(variable_name_type_dimension_to_complexity_group_dict.values()) == 0:
        return None, False, None, None, None

    complexity_group_list = list(set(variable_name_type_dimension_to_complexity_group_dict.values()))

    if print_info:
        print('number of resulting groups', len(complexity_group_list))

    if len(set(complexity_group_list)) == 1:
        main_complexity_group = complexity_group_list[0]
    else:
        main_complexity_group = GroupComplexity()
        main_complexity_group.print_info = print_info
        for group_or_node in set(complexity_group_list):
            main_complexity_group.add_group_or_node(group_or_node)

    if print_info:
        print('below the tree')
        print(main_complexity_group.get_encapsulated_variable_name_type_dimension())

    main_complexity_group.self_set_complexity(
        max_time=None,
        fix_negligeable_complexity=fix_negligeable_complexity,
        max_time_rate=max_time_rate,
        elect_function=elect_function,
        fix_constant_complexity=fix_constant_complexity,
    )

    main_complexity_group.self_adjust_group_operations(
        max_time=None,
        fix_negligeable_complexity=fix_negligeable_complexity,
        max_time_rate=max_time_rate,
        elect_function=elect_function,
        fix_constant_complexity=fix_constant_complexity,
    )

    main_complexity_group.self_adjust_constant_complexity(
        max_time=None,
        fix_negligeable_complexity=fix_negligeable_complexity,
        max_time_rate=max_time_rate,
        elect_function=elect_function,
        fix_constant_complexity=fix_constant_complexity,
    )

    main_complexity_group.self_adjust_group_operations(
        max_time=None,
        fix_negligeable_complexity=fix_negligeable_complexity,
        max_time_rate=max_time_rate,
        elect_function=elect_function,
        fix_constant_complexity=fix_constant_complexity,
    )

    if print_info:
        plt.legend()

    formatted_complexity = main_complexity_group.format_complexity(
        letter_list=['n', 'm', 'k', 'l', 'u', 'v', 'w'],
        next_letter_index=0,
    )

    any_peak = None
    if len(found_peak_max_time_list):
        max_time_ref = max(list(map(lambda x: x[1], found_peak_max_time_list)))
        any_peak = bool(
            sum(list(map(lambda x: int(x[0]), list(filter(lambda x: x[1] >= max_time_ref * 0.8, found_peak_max_time_list))))) > 0
        )

    return 'o({})'.format(formatted_complexity[0]), False, formatted_complexity[2], formatted_complexity[3], any_peak