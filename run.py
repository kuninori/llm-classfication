import utils
import torch
from model import Model
from transformers import LlamaTokenizer, AutoTokenizer

taglist = utils.read_taglist()
model = Model(num_labels=len(taglist))
model.load_state_dict(torch.load("model.prod.pth"))
model.eval()

tokenizer_name = "novelai/nerdstash-tokenizer-v1"
# tokenizer_name = "japanese-gpt-neox-3.6b";
# model_name = "japanese-gpt-neox-3.6b";

# tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, fas)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
inputs = tokenizer(
    text="""楽天モバイル、フラッグシップスマホ「Galaxy S23 SM-S911C」にAndroid 14へのOSバージョンアップを含むソフトウェア更新が提供開始
Rakuten Mobileスマホ「Galaxy S23 SM-S911C」がAndroid 14に！
楽天モバイルは5日、同社が移動体通信事業者（MNO）として自社回線（以下、楽天回線）を構築して提供している携帯電話サービス（ https://network.mobile.rakuten.co.jp/ ）に対応している楽天回線対応製品として販売している5G対応フラッグシップスマートフォン（スマホ）「Galaxy S23（型番：SM-S911C）」（Samsung Electronics製）に対して最新プラットフォーム「Android 14」へのOSバージョンアップを含むソフトウェア更新を2023年12月5日（火）より順次提供開始したとお知らせしています。
更新はスマホ本体のみで無線LAN（Wi-Fi）によるネットワーク経由（OTA）のほか、パソコン（PC）に接続して実施する方法が用意されており、更新後のビルド番号は「UP1A.231005.007.S911CONU1BWK2」。なお、更新時間や更新ファイルサイズは明らかにされていません。主な更新内容は「OSアップデート（Android 14）」となっていますが、合わせて独自ユーザーインターフェース「One UI 6.0」にアップデートされるとのことで、One UI 6.0についてはGalaxyの公式Webページ『One UI 6の新機能 | Samsung Japan 公式 JP』にてご確認ください。""".strip(),
    add_special_tokens=False,
    truncation=True,
    padding=True,
    return_tensors="pt",
)

output = model(**inputs)
pred_labels = output.logits.argmax(dim=-1).tolist()
print(pred_labels, [taglist[i] for i in pred_labels])
