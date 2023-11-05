import torch
from tqdm import trange, tqdm
from torch.optim import AdamW
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

dataset = []
dataloader = []
num_steps = len(dataloader)
model = Model()
print(model.named_parameters())
# dictをtensorに変換する。textとlabelsにする
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

def train(model:Model):
    model.train()

    optimizer = AdamW(model.named_parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_steps,
        num_training_steps=num_steps * epochs
    )

    for epoch in trange(epochs, dynamic_ncols=True):
        for batch in tqdm(dataloader, dynamic_ncols=True):
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            model.backward(loss)

            optimizer.step()
            lr_scheduler.step()
        model.eval()



def evaluate():
    pass