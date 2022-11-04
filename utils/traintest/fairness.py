import torch
import numpy as np
import utils.traintest.evaluation as ev


def get_output_and_protected_label(
    model, device, loader, attribute_name="Male", max_batches=20000
):
    out = []
    gt_y = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            out.append(model(x.to(device)).detach().cpu())
            gt_y.append(y)
            if idx >= max_batches:
                break
    output = torch.cat(out, 0)
    gt_y = torch.cat(gt_y, 0)
    return output, gt_y, loader.dataset.df[attribute_name].values


def get_fairness_metrics(model, device, loader, attribute_name="Male"):
    model.eval()
    output, ground_truth, protected = get_output_and_protected_label(
        model, device, loader, attribute_name=attribute_name
    )

    subset_id = protected == 1
    output_a, output_b = output[subset_id], output[~subset_id]
    ground_truth_a, ground_truth_b = ground_truth[subset_id], ground_truth[~subset_id]

    fairness_dict = {"Group A": {}, "Group B": {}, "Difference": {}}

    for metric in metric_dict:
        outcome_a = metric_dict[metric](output_a, ground_truth_a).item()
        outcome_b = metric_dict[metric](output_b, ground_truth_b).item()
        fairness_dict["Group A"][metric] = outcome_a
        fairness_dict["Group B"][metric] = outcome_b
        fairness_dict["Difference"][metric] = outcome_a - outcome_b

    return fairness_dict


def accuracy(output, ground_truth):
    return (output.max(1)[1] == ground_truth).float().mean()


def acceptance_rate(output, ground_truth):
    pred = output.max(1)[1]
    return pred.float().mean()


def fpr(output, ground_truth):
    pred = output.max(1)[1]
    fp = ((pred) * (1 - ground_truth)).sum()
    tn = ((1 - pred) * (1 - ground_truth)).sum()
    return fp / (fp + tn)


def fnr(output, ground_truth):
    pred = output.max(1)[1]
    fn = ((1 - pred) * (ground_truth)).sum()
    tp = ((pred) * (ground_truth)).sum()
    return fn / (fn + tp)


def tpr(output, ground_truth):
    pred = output.max(1)[1]
    tp = ((pred) * (ground_truth)).sum()
    fn = ((1 - pred) * (ground_truth)).sum()
    return tp / (tp + fn)


def ppv(output, ground_truth):
    pred = output.max(1)[1]
    tp = ((pred) * (ground_truth)).sum()
    fp = ((pred) * (1 - ground_truth)).sum()
    return tp / (tp + fp)


def prevalence(output, ground_truth):
    return (ground_truth).sum() / len(ground_truth)


metric_dict = {
    "acc": accuracy,
    "acceptance_rate": acceptance_rate,
    "fpr": fpr,
    "fnr": fnr,
    "tpr": tpr,
    "ppv": ppv,
    "prevalence": prevalence,
}


def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def return_pareto_frontier(model, loader, device, attribute_name="Male"):
    output, ground_truth, protected = get_output_and_protected_label(
        model, device, loader, attribute_name=attribute_name
    )

    subset_id = protected == 1
    ground_truth_a, ground_truth_b = ground_truth[subset_id], ground_truth[~subset_id]
    biases = torch.linspace(-1.5, 1.5, 20)

    results = []

    for male_bias in biases:
        for female_bias in biases:
            bias_shift = torch.zeros_like(output)
            bias_shift[subset_id, 1] = male_bias
            bias_shift[~subset_id, 1] = female_bias

            biased_output = output + bias_shift

            output_a, output_b = biased_output[subset_id], biased_output[~subset_id]

            acc = accuracy(biased_output, ground_truth).item()
            fnr_men = fnr(output_a, ground_truth_a).item()
            fnr_women = fnr(output_b, ground_truth_b).item()
            fnr_diff = fnr_men - fnr_women
            results.append((acc, fnr_diff))
    results = np.array(results)

    transformed_results = results.copy()
    transformed_results[:, 0] = -transformed_results[:, 0]
    transformed_results[:, 1] = np.abs(transformed_results[:, 1])
    pareto_results = results[is_pareto_efficient_simple(transformed_results)]
    return pareto_results
