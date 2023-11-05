import peft
from peft import LoraConfig, PeftModel
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from transformers import LlamaTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutput

lora_r = 32
num_labels = 10
batch_size = 32
epochs = 10
max_seq_len = 128
model_name = "stabilityai/japanese-stablelm-base-alpha-7b",
tokenizer_name = "novelai/nerdstash-tokenizer-v1"

tokenizer = LlamaTokenizer.from_pretrained(
    tokenizer_name,
    additional_special_tokens=['▁▁'],
    max_seq_len=max_seq_len,
    use_fast=True,
)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            load_in_8bit=True,
        )
        self.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=16,
            lora_dropout=0.1,
            inference_mode=False,
            target_modules=["embed_out"],
        )
        self.backbone: PeftModel = peft.get_peft_model(backbone, self.peft_config)
        self.backbone.enable_input_require_grads()
        self.backbone.gradient_checkpointing_enable()

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, labels: LongTensor):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq_length = attention_mask.sum(dim=1)
        eos_hidden_states = outputs.last_hidden_state(
            torch.arange(
                seq_length.size(0),
                device=outputs.last_hidden_state.device,
            ),
            seq_length - 1,
        )
        logits = self.classifier(eos_hidden_states)
        loss = self.loss_fn(logits, labels)
        return SequenceClassifierOutput(loss, logits)

device = torch.device("cuda:0")
model = Model()
# model.to(device)
model.eval()

prompt = """
AI で科学研究を加速するには
""".strip()
input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt", )

seed = 23
torch.manual_seed(seed)
tokens = model.backbone.generate(
    input_ids=input_ids.to(device=device),
    max_new_tokens=256,
    temperature=1,
    top_p=0.95,
    do_sample=True,
).to(device)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)
