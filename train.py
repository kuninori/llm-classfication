import torch
from tqdm import trange, tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from datasets import TextDataset
from model import Model
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import LlamaTokenizer, AutoModelForCausalLM, BatchEncoding


lr = 5e-4
max_seq_len = 512
seed = 42
lora_r = 32
num_labels = 10
batch_size = 32
epochs = 10
model_name = "stabilityai/japanese-stablelm-base-alpha-7b",
tokenizer_name = "novelai/nerdstash-tokenizer-v1"

tokenizer = LlamaTokenizer.from_pretrained(
    pretrained_model_name_or_path=tokenizer_name,
    additional_special_tokens=['▁▁'],
    max_seq_len=max_seq_len,
    use_fast=True,
)

def collate_fn(datalist) -> BatchEncoding:
    inputs = tokenizer(
        text= [d["text"] for d in datalist],
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=max_seq_len,
    )

    labels = torch.LongTensor([d["labels"] for d in datalist])
    return BatchEncoding({ **inputs, "labels":labels })

def create_dataloader(dataset):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    num = len(dataset)
    train = num * 0.8
    val = num - train

    return random_split(dataset, [train, val])

dataset = TextDataset()
train_dataloader, val_dataloader = create_dataloader(dataset)
print(train_dataloader, val_dataloader)
num_steps = len(train_dataloader)
model = Model()
print([name for name, param in model.named_parameters()])
# dictをtensorに変換する。textとlabelsにする
def train(model:Model, dataloader):
    model.train()

    best_val_f1 = 0
    best_state_dict = model.state_dict()
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_steps,
        num_training_steps=num_steps * epochs
    )

    for epoch in trange(epochs, dynamic_ncols=True):
        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True):
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            model.backward(loss)

            optimizer.step()
            lr_scheduler.step()

        model.eval()
        val_outputs = evaluate(dataloader)

        if  val_outputs["f1"] > best_val_f1:
            best_val_f1 = val_outputs["f1"]
            best_state_dict = model.state_dict()
    model.load_state_dict(best_state_dict)
    model.eval()





def evaluate(dataloader):
    pass


train(model, train_dataloader)

