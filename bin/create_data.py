import csv
import json

import openai

API_KEY = "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"
openai.api_key = API_KEY

filename = "k10014242391000"

tags = []
with open("./tags.json", mode="r") as f:
    tags = json.load(f)


def save_text(text):
    with open(f"./nhk_news/texts/{filename}.txt", "w") as f:
        f.write(text)


def create_data(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"""
            あなたはニュース編集者です。
            """},
            {"role": "user", "content": f"""
            「{text}」というテキストに関連度が高いタグとその関連度のスコア(0-1)を10個返してください。
            結果をそのままJSONとして使用したいので不要な文字は含めないでください。
            JSONの構造は`[[タグ1, スコア1],[タグ2, スコア2]]`の形としてください

            タグは以下の中から必ず選んでください。それぞれ関連度をスコア化して0-1の値で返してください。
            内容についてタグをつけてください。ニュースなどのタグは使用しないでください。
            {tags}
            """}
        ]
    )
    save_text(text)
    res = response["choices"][0]["message"]["content"]
    return json.loads(res)


res = create_data("""
[タイトル]
G7外相会合 7日から東京で開催 イスラエル パレスチナ情勢議論

[本文]
G7＝主要7か国の外相会合が、7日から2日間東京で開かれ、初日はイスラエル・パレスチナ情勢について議論が交わされる見通しです。議長国の日本としては、人道目的の一時的な戦闘休止の必要性などを訴え、G7で結束したメッセージを打ち出せるよう議論をリードしていく考えです。

G7外相会合は、東京 港区の飯倉公館で7日と8日の2日間開かれます。

イスラエルとハマスの衝突が始まってから、7日で1か月ですが、G7外相が一堂に会するのは初めてで、最初の会合となるワーキングディナーでは、緊迫するイスラエル・パレスチナ情勢が主な議題となる見通しです。

会合では上川外務大臣が議長を務め、ハマスなどによるテロ攻撃を非難するとともに、イスラエルには自国や自国民を守る権利があるという認識を共有するものとみられます。

一方で日本としては、ガザ地区で民間人の犠牲者が増え続ける中、人道目的の一時的な戦闘の休止や国際法を順守する必要性などを訴える方針です。

そして、G7が結束して事態の沈静化や人道状況の改善につながるメッセージを打ち出せるよう議論をリードしていく考えです。

""")

with open(f"./nhk_news/tags/{filename}.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(res)
