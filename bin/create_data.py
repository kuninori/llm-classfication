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
パナソニックHD 半年間決算 最終利益2883億円で過去最高

パナソニックホールディングスのことし4月から9月までの半年間の決算は、EV＝電気自動車向けの電池の販売が好調だったことなどから最終的な利益が過去最高となる2883億円となりました。

パナソニックホールディングスが発表したことし4月から9月までの半年間のグループ全体の決算は
▽売り上げが前の年の同じ時期から1.4％増えて4兆1194億円
▽最終的な利益は前の年の同じ時期から2.6倍の2883億円でした。

最終的な利益はこの時期としては過去最高を更新しました。

中国でのパソコンやスマートフォン需要の低迷などで電子部品事業が苦戦を強いられたものの、
▽アメリカのEVメーカー・テスラ向けなどの電池の販売が好調だったことや
▽アメリカの工場でのEV向けの電池生産に伴ってアメリカ政府などから補助金が見込まれていることが主な要因です。

さらに、円安による業績の押し上げ効果もあったということです。

梅田博和グループCFOは、オンラインの記者会見で「強い事業と苦戦した事業がくっきり分かれた決算になった。今後はメリハリをつけて体質強化を図っていきたい」と述べました。

そのうえで、今後の事業再編について「議論をして方向性を決めようとしているところだ。しかるべきタイミングで報告できると思う」と述べ、検討を具体的に進めていることを明らかにしました。
""")

with open(f"./nhk_news/tags/{filename}.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(res)
