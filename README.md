# arXiv-search-local
タイトルとアブストラクトの情報から類似するarXivの論文を検索するツールです。TransformerのPretrainedモデルである[SPECTER](https://arxiv.org/abs/2004.07180) [Cohan+, ACL 2020]により論文のタイトルとアブストラクトの埋め込みベクトルを計算し、その類似度を利用して類似論文を検索します。SPECTERは引用関係にある論文の出力ベクトル同士が類似するように学習を行っているため、推論時には引用関係にあると推定される論文間の距離が近くなるようにベクトルを出力します。

**本ツールは下記URLでwebアプリケーションとして公開しています。**  
https://share.streamlit.io/tetsu9923/arxiv-search-streamlit/main.py


## requirements
- torch
- numpy 
- transformers
- requests
- feedparser==5.1.1

Ubuntu Server 16.04, Python3.8で動作確認済み


## 手順
以下の手順で論文検索を行います。
1. arXiv APIを利用した論文のダウンロード
2. SPECTERによる論文埋め込みベクトルの生成（GPU使用推奨）
3. クエリ論文を与えて類似論文検索

ここで、2021年までに公開された機械学習系（カテゴリ: cs.AI, cs.LG, stat.ML, cs.CV, cs.CL）の論文データについては[こちら](https://drive.google.com/drive/folders/18TD8I6T8sTLtjngR1mIc7PVafb3sOkmA?usp=sharing)で公開しているため、arXiv-search-local/data/ディレクトリを作り、data.tar.gzを解凍した中身のデータを格納すれば、3.の手順のみで類似論文を検索できます。

## 論文のダウンロード
まず、arXivのAPIを利用して論文データをダウンロードします。以下のコマンドはカテゴリcs.AIもしくはcs.CVに属する論文のうち、過去10日間にpublishされた論文をダウンロードする例です。
```
python get-papers.py --query cat:cs.AI+OR+cat:cs.CV --day-minus 10
```
各argsの詳細は以下の通りです（全てoptional）。公式ドキュメントの[Query Interface](https://arxiv.org/help/api/user-manual#_query_interface)も参考にしてください。
|  --option  | 説明 | Default |
|-|-|-|
| --query | arXiv APIのクエリ（詳細は[こちら](https://arxiv.org/help/api/user-manual#query_details)を参照してください） | cat:cs.LG |
| --day-minus | 何日前までの論文をダウンロードするかを示す。全期間の論文をダウンロードするには10000と指定してください。 | 10000 |
| --append | 今回ダウンロードした論文データを前回までにダウンロードしたデータに付加するかどうかを示す。--appendを付けなければ上書きされます。 | (False) |
| --max-results | 1回のリクエストでダウンロードする論文数の最大値。10000以下の値に設定してください。値が大きいとリクエストに繰り返し失敗することがあるので適宜調整してください。詳しくは[Query Interface](https://arxiv.org/help/api/user-manual#_query_interface)を参照してください。 | 10000 |
| --start-idx | 何番目の論文からダウンロードするかを表すindex。詳しくは[Query Interface](https://arxiv.org/help/api/user-manual#_query_interface)を参照してください。 | 0 |


## SPECTERによる論文埋め込みベクトルの生成
次に、[SPECTER](https://arxiv.org/abs/2004.07180) [Cohan+, ACL 2020]による論文タイトル・アブストラクトの埋め込みベクトルを生成します。
```
python embed.py --device cuda:0 --batch-size 32
```
各argsの詳細は以下の通りです（全てoptional）。
|  --option  | 説明 | Default |
|-|-|-|
| --device | デバイスを示す。GPUを使用する場合は`cuda:x`、CPUを使用する場合は`cpu`と指定してください。 | cuda:0 |
| --batch-size | バッチサイズ | 32 |


## 類似論文検索
最後に、クエリとなるタイトルもしくはアブストラクト（もしくはその両方）を指定して類似論文検索を行います。結果は標準出力と`./data/results.txt`に出力されます。
```
python similar-search.py --top-n 10 --title "title of your paper" --abstract "abstract of your paper"
```
各argsの詳細は以下の通りです（全てoptional）。
|  --option  | 説明 | Default |
|-|-|-|
| --top-n | 類似度の上位何件の結果を表示するかを示す。 | 10 |
| --title | 検索したいタイトル | None |
| --batch-size | 検索したいアブストラクト | None |
