import torch
from autoattack import AutoAttack


threat_list = [("Linf", 0.01), ("L2", 1.0), ("L1", 20)]


def get_adversarial_metrics(model, device, loader):
    images = []
    labels = []
    for x, y in loader:
        images.append(x)
        labels.append(y)
    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)

    adversarial_dict = {}
    for norm, eps in threat_list:
        adversary = AutoAttack(model, norm=norm, eps=eps, version="standard")
        adversary.attacks_to_run = ["apgd-ce"]
        x_adv, y_adv = adversary.run_standard_evaluation(
            images, labels, bs=loader.batch_size, return_labels=True
        )
        rob_acc = (labels == y_adv).float().mean().item()
        adversarial_dict[norm + "_" + str(eps)] = rob_acc

    return adversarial_dict
