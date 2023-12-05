import utils
import torch
from model import Model
from transformers import LlamaTokenizer

taglist = utils.read_taglist()
model = Model(num_labels=len(taglist))
model.load_state_dict(torch.load("model.prod.pth"))
model.eval()

tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
inputs = tokenizer(
    text=""" """.strip(),
    add_special_tokens=False,
    truncation=True,
    padding=True,
    return_tensors="pt",
)

print(inputs.input_ids.shape)

output = model(**inputs)
pred_labels = output.logits.argmax(dim=-1).tolist()
print(pred_labels, [taglist[i] for i in pred_labels])
