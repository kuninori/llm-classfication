import utils

import torch
from tqdm import trange, tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from livedoor_datasets import LivedoorDataset
from model import Model
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import LlamaTokenizer, AutoModelForCausalLM, BatchEncoding, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from accelerate import Accelerator
from config import model_name, tokenizer_name

lr = 1e-3
max_seq_len = 128
seed = 22
batch_size = 2
epochs = 12

accelerator = Accelerator(cpu=False)
taglist = utils.read_taglist()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

def collate_fn(datalist) -> BatchEncoding:
    inputs = tokenizer(
        text=[text for (text, _) in datalist],
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=max_seq_len,
    )
    labels = []
    for _, tag in datalist:
        labels.append(taglist.index(tag))
    labels = torch.LongTensor(labels)
    return BatchEncoding({**inputs, "labels": labels})


def dataloaders():
    dataset = LivedoorDataset()
    all_num = len(dataset)
    train_num = int(all_num * 0.8)
    val_num = int(all_num - train_num)
    train_dataset, val_dataset = random_split(dataset, [train_num, val_num])
    train_dataloader = create_dataloader(train_dataset)
    val_dataloader = create_dataloader(val_dataset)

    return train_dataloader, val_dataloader


def create_dataloader(dataset):
    return DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )


train_dataloader, val_dataloader = dataloaders()
num_steps = len(train_dataloader)
model = Model(num_labels=len(taglist))
model.backbone.print_trainable_parameters()

# dictをtnsorに変換する。textとlabelsにする
def train(model: Model, train_dataloader, val_dataloader):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    best_val_loss = 0
    best_state_dict = model.state_dict()
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_steps,
        num_training_steps=num_steps * epochs
    )

    model, train_dataloader, val_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
    )

    for epoch in trange(epochs, dynamic_ncols=True):
        for batch in tqdm(train_dataloader, total=len(train_dataloader), dynamic_ncols=True):
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss

                before = torch.clone(model.classifier.weight)
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        model.eval()
        (loss, accuracy, f1, precision, recall) = evaluate(model, val_dataloader)
        print(f"loss:{loss}, accuracy:{accuracy}, f1:{f1}, precision:{precision}, recall:{recall}")

        if loss < best_val_loss:
            best_val_loss = loss
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, "model.pth")

    model.load_state_dict(best_state_dict)
    model.eval()

    torch.save(best_state_dict, "model.pth")

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    val_labels = []
    pred_labels = []

    for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
        output = model(**batch)
        batch_size = batch.input_ids.size(0)
        loss = output.loss.item() * batch_size
        pred_labels += output.logits.argmax(dim=-1).tolist()
        val_labels += batch.labels.tolist()
        total_loss += loss

    loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(pred_labels, val_labels)
    print(total_loss, val_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_labels,
        pred_labels,
        average="macro",
        zero_division=0,
    )
    return (loss, accuracy, f1, precision, recall)

train(model, train_dataloader, val_dataloader)
