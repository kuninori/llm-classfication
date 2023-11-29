import peft
from peft import LoraConfig, PeftModel
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from transformers import LlamaTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutput

lora_r = 4
max_seq_len = 128
model_name = "stabilityai/japanese-stablelm-base-alpha-7b",

print("bf16:", torch.cuda.is_bf16_supported())

class Model(nn.Module):
    def __init__(self, num_labels):
        super(Model, self).__init__()
        self.num_labels = num_labels
        backbone = AutoModelForCausalLM.from_pretrained(
            # "stabilityai/japanese-stablelm-3b-4e1t-base",
            "stabilityai/japanese-stablelm-base-alpha-7b",
            use_cache=False,
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None,
        )
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            lora_dropout=0.1,
            inference_mode=False,
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
            target_modules=[
                "query_key_value",
                "dense",
                "packed_input_proj",
                "out_proj",
            ],
        )
        self.backbone: PeftModel = peft.get_peft_model(backbone, self.peft_config)
        self.backbone.enable_input_require_grads()
        from functools import partial
        notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        torch.utils.checkpoint.checkpoint = notfailing_checkpoint
        self.backbone.gradient_checkpointing_enable()

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(self.classifier.weight, std=0.01)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, labels: LongTensor):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq_length = attention_mask.sum(dim=1)
        last_hidden_state = outputs.hidden_states[-1]
        eos_hidden_states = last_hidden_state[
            torch.arange( seq_length.size(0), device=last_hidden_state.device),
            seq_length - 1,
        ]
        logits = self.classifier(eos_hidden_states)
        loss = self.loss_fn(logits, labels)
        return SequenceClassifierOutput(loss, logits)


if __name__ == "__main__":
    tokenizer_name = "novelai/nerdstash-tokenizer-v1"
    tokenizer = LlamaTokenizer.from_pretrained(
        tokenizer_name,
        additional_special_tokens=['▁▁'],
        max_seq_len=max_seq_len,
        use_fast=True,
    )
    device = torch.device("cuda:0")
    model = Model()
    model.eval()

    prompt = """
    AI で科学研究を加速するには
    """.strip()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

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
