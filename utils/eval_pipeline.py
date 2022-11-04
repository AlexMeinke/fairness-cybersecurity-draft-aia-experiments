import yaml
import torch
from tinydb import TinyDB
import utils.traintest.evaluation as ev
import utils.dataloaders.dataloading as dataloading
import utils.factories as fac
import paths_config
from utils.factories import dotdict
import numpy as np
import utils.traintest.fairness as fair
from utils.adversarial.adversarial_eval import get_adversarial_metrics


def gen_eval(doc_id, device=torch.device("cuda:0")):
    ##### Access database #####
    db = TinyDB(paths_config.project_folder + "evals/CelebA.json")
    data = {}

    entry = db.get(doc_id=doc_id)
    train_args = dotdict(entry["args"])
    for sub in train_args:
        if type(train_args[sub]) == dict:
            train_args[sub] = dotdict(train_args[sub])

    ##### Get Loaders #####
    if "batch_size" in train_args["eval"]:
        batch_size = train_args["eval"]["batch_size"]
    else:
        batch_size = 100
    train_args["eval"]["batch_size"] = batch_size

    in_loader, loaders_out = fac.get_test_loaders(train_args)

    ##### Get Model #####
    model = fac.get_model(dotdict(train_args["architecture"])).to(device)
    model.eval()
    arch_style = train_args["architecture"]["arch_style"]
    if arch_style.lower() == "sep":
        arch_style = "joint"

    ##### ID Stats #####
    num_classes = train_args["architecture"]["num_classes"]
    from_logits = (arch_style.lower() in ["rn", "cnn", "dn"]) and num_classes != 1
    use_last_class = arch_style.lower() == "densenet"

    if num_classes != 1:
        acc, conf = ev.get_accuracy(model, device, in_loader, from_logits=from_logits)
        data["acc"] = acc
        data["MMC"] = conf.mean().item()
        if use_last_class:
            conf = -torch.log_softmax(ev.get_output(model, device, in_loader), dim=1)[
                :, -1
            ]
    else:
        conf = torch.sigmoid(ev.get_output(model, device, in_loader).squeeze())

    ##### Fairness Stats #####
    data["Fairness"] = fair.get_fairness_metrics(model, device, in_loader)

    ##### Adv Robustness Stats #####
    data["Adversarial"] = get_adversarial_metrics(model, device, in_loader)

    db.update({"results": data}, doc_ids=[doc_id])
    return data
