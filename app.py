import pandas as pd
import numpy as np
import streamlit as st
import requests
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px

# API呼び出しの結果を格納するためのセッションステートの初期化
if 'api_result' not in st.session_state:
    st.session_state.api_result = None


# Streamlitのタイトル
st.sidebar.title('テキスト感情分析')

# 見本用のテキスト
selected_item = st.sidebar.selectbox('サンプルテキスト', 
    [
    '',
    'サンプルテキスト',
    '商品レビュー',
    'トラブル',
    '吾輩は猫である', 
    '草枕', 
    'ポラーノの広場'
    ],
    help='リストからテキストのセットを呼び出します')

aozora_dict = {
    'サンプルテキスト':"サンプルテキストです。雨が降って気分がどんより。こんな時こそ笑顔が一番！空が晴れたら虹もかかった。素晴らしい一日になりました。",
    '商品レビュー':"このリップクリーム、すっごく良かったです！　唇がプルプルになりました！使用感も軽やかで、塗り心地は本当に滑らか。色持ちも良くて、朝塗れば夕方までしっかり色が残っているんです。それに、このリップクリームの香り！ほんのり甘くて、つけているだけで気分が上がります。包装もシンプルでエレガント、持ち歩くのが嬉しくなるデザイン。でも、ちょっと高すぎ。確かに品質はいいけど、他のブランドのリップクリームと比較すると、かなり高価。毎日使うものと考えると少し手が出しづらい価格設定かもしれません。あと、種類が豊富すぎて選びづらいという問題も。どれが自分に合っているのか迷ってしまいます。それでも、このリップクリームは最近の一押しです。まだ買ったことのない人は一度試してみることをオススメします。",
    'トラブル':"看板にソフトクリーム屋とあるのに、アイスが全くないのはちょっと驚きました。正直、期待していた分、がっかりはしましたね。でも、店員さんも困っている様子で、マシンが壊れてしまったらしい。そういうアクシデントはどこにでもあることだし、仕方ないのかなとも思います。ただ、せめて入店前に何らかの告知があれば、もう少し気持ちの整理もついたかもしれません。店員さんの対応は、状況を踏まえれば、まぁ頑張っていた方かなと。今回は満足いく体験ではなかったですが、修理が済んだら、また改めて足を運んでみようかな、とは思います。次こそは、期待に応えてくれるといいなと期待しています。",
    '吾輩は猫である':"吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。", 
    '草枕':"山路を登りながら、こう考えた。智に働けば角が立つ。情に棹させば流される。意地を通せば窮屈だ。とかくに人の世は住みにくい。住みにくさが高じると、安い所へ引き越したくなる。どこへ越しても住みにくいと悟った時、詩が生れて、画が出来る。", 
    'ポラーノの広場':"そのころわたくしは、モリーオ市の博物局に勤めて居りました。十八等官でしたから役所のなかでも、ずうっと下の方でしたし俸給もほんのわずかでしたが、受持ちが標本の採集や整理で生れ付き好きなことでしたから、わたくしは毎日ずいぶん愉快にはたらきました。殊にそのころ、モリーオ市では競馬場を植物園に拵こしらえ直すというので、その景色のいいまわりにアカシヤを植え込んだ広い地面が、切符売場や信号所の建物のついたまま、わたくしどもの役所の方へまわって来たものですから、わたくしはすぐ宿直という名前で月賦で買った小さな蓄音器と二十枚ばかりのレコードをもって、その番小屋にひとり住むことになりました。わたくしはそこの馬を置く場所に板で小さなしきいをつけて一疋の山羊を飼いました。毎朝その乳をしぼってつめたいパンをひたしてたべ、それから黒い革のかばんへすこしの書類や雑誌を入れ、靴もきれいにみがき、並木のポプラの影法師を大股にわたって市の役所へ出て行くのでした。あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。"
    }

sample_text = "リストから選択してください"
if selected_item in aozora_dict:
    sample_text = aozora_dict[selected_item]

# ユーザー入力用のテキストエリア→サンプルは文字表示のみに変更
st.sidebar.text("テキスト内容")
container = st.sidebar.container(border=True)
container.write(str(sample_text)[:100] + "…" if len(sample_text)>100 else sample_text)

# サンプルアプリなのでkeyを渡す
text = selected_item

# ボタンウィジェット
if st.sidebar.button("分析開始"):
    if text != "":
        # FastAPIサーバーへのリクエスト
        response = requests.post("http://localhost:8000/analyze", json={"text": text})
        
        if response.status_code == 200:
            st.session_state.api_result = response.json()
            st.session_state.api_called = True
        else:
            st.error("APIエラーが発生しました。")
    else:
        st.info("テキストを入力してください。")

# APIの結果を表示
if st.session_state.api_result:

    df = pd.DataFrame(st.session_state.api_result)
    
    # 感情カテゴリ
    positive = ['喜び', '期待']
    negative = ['悲しみ', '怒り']
    natural_positive = ['驚き']
    natural_negative = ['恐れ']
    
    # 感情リストを展開
    emo_df = pd.DataFrame()
    for index, row in df.iterrows():
        for emotion, score in row['scores'].items():
            emo_df.at[index, emotion] = score

    emo_df = emo_df.reindex(columns=['喜び', '期待', '驚き', '恐れ', '悲しみ', '怒り'])
    
    # ポジティブかネガティブかで重さを計算する関数
    def calculate_sentiment_score(row):
        score = 0.5
        for emotion in positive:
            score += row.get(emotion, 0)
        for emotion in negative:
            score -= row.get(emotion, 0)
        for emotion in natural_positive:
            score += row.get(emotion, 0) / 2
        for emotion in natural_negative:
            score -= row.get(emotion, 0) / 2
        return score
    
    # dfに感情カラー用重み付きスコアを追加
    weighted = emo_df.apply(calculate_sentiment_score, axis=1)*2
    df['normalized_scores'] = (weighted - weighted.min()) / (weighted.max() - weighted.min())
    
    
    #出力用データテーブルの作成
    st_df = pd.concat([df,emo_df],axis=1)
    st_df = st_df.drop(labels=['max_score','scores'],axis=1)
    
    st_df.rename(
        columns={
            'sentence': '文章',
            'sentiment': 'テンション',
            'max_label': '感情',
            'normalized_scores': 'ポジティブ度'
            },
        inplace=True
        )
    
    # データフレームをCSV形式→バイト形式に変換
    csv = st_df.to_csv(index=False)
    to_write = StringIO(csv)
    
    # プロットの用意
    fig = go.Figure()
    
    # x軸ラベルの作成
    def create_label(row):
        x_label = str(row.name) + "　" + row['sentence'][:8]
        if len(row['sentence']) > 8:
            x_label = x_label + "…"
        return x_label
    sentence_label = df.apply(create_label, axis=1)
    
    # 折れ線チャートの追加
    fig.add_trace(go.Scatter(
        x=sentence_label, 
        y=df['sentiment'],
        mode='lines',
        line=dict(color='black'),
        showlegend=False
    ))
    
    # 分散図のホバーテキストの生成
    hover_texts = df.apply(lambda row: 
        f"{row['sentence'][:50]}<br>" +
        f"感情: {row['max_label']}" +
        {"喜び": "😄", "期待": "🙂", "驚き": "😲", "恐れ": "😨", "悲しみ": "😢", "怒り": "😠"}.get(row['max_label'], "") +
        "　" +
        f"大きさ: {round(row['max_score']*100,1)}%<br>",
        axis=1
    ).tolist()
    
    # カラースケールを定義
    color_scale = px.colors.sequential.Viridis
    
    # カラースケールに合わせて重み付きスコアを調整
    df.loc[(df['normalized_scores'] < 0.3) & (df['max_label'].isin(positive)), 'normalized_scores'] = 0.5
    
    # 散布図の追加
    fig.add_trace(go.Scatter(
        x=sentence_label, y=df['sentiment'], 
        mode='markers',
        marker=dict(
            size=df['max_score']*30, 
            color=df['normalized_scores'], 
            colorscale=color_scale,
            showscale=True,
            coloraxis='coloraxis',  # カラーアクシスの参照を追加
            ),
        text=hover_texts,  # ホバーテキストを設定
        hoverinfo='text'  # テキストを表示
    ))
    
    # レイアウトの設定
    fig.update_layout(
        title='感情折れ線チャート',
        xaxis_title='文章',
        xaxis=dict(
            tickangle=90
            ),
        yaxis_title='テンション',
        yaxis=dict(
            range=(-3, 3)
            ),
        coloraxis=dict(colorscale=color_scale),
        showlegend=False,
        height=600,
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=1.1,  # カラーバーの直ぐ隣を指す
                y=1,  # カラーバーの上部
                text="Positive",
                showarrow=False,
                align="left"
            ),
            dict(
                xref='paper',
                yref='paper',
                x=1.1,  # カラーバーの直ぐ隣を指す
                y=0.15,  # カラーバーの下部
                text="Negative",
                showarrow=False,
                align="left"
            )
        ]
    )

    # Streamlitで表示
    st.header('分析結果')
    
    def to_percentage(x):
        if isinstance(x, float):
            return "{:.1f}%".format(x * 100)
        return x
    
    def ActiveSentiment(sentiment):
        list_size = len(sentiment)
        # 最大値(+2)と最小値(-2)を均等に分配
        # 要素数が奇数の場合、中央の値を0に設定（平均値に近づけるため）
        if list_size % 2 == 0:
            # 偶数の場合、半分を最小値、半分を最大値にする
            scores = [-2] * (list_size // 2) + [2] * (list_size // 2)
        else:
            # 奇数の場合、最小値と最大値の間に1つ0を挿入
            scores = [-2] * (list_size // 2) + [0] + [2] * (list_size // 2)
        return np.std(sentiment)/np.std(scores)
    
    # ポジティブ度の説明文dict
    mentality_descriptions = {
        1 : "全面的に悲観的です。",
        2 : "否定的であり、悲観的な見方が強く出ています。",
        3 : "ほぼ否定的で、楽観的要素が少ないです。",
        4 : "悲観的な見方がありつつ、少し希望が見えます。",
        5 : "中立的な立場を保っている文章です。",
        6 : "ポジティブな見方が感じられます。",
        7 : "明るく希望的な内容が含まれています。",
        8 : "非常にポジティブな視点を持っています。",
        9 : "極めて前向きで、明るい内容です。"
    }

    # テンション度の説明文dict
    tension_descriptions = {
        1 : "動きがなく、完全に静かな状態です。",
        2 : "非常に落ち着いており、感情の起伏がありません。",
        3 : "やや低めのテンションで、感情の起伏は穏やかです。",
        4 : "穏やかな動きで、落ち着いた雰囲気です。",
        5 : "普通の活動レベルで、バランスが取れています。",
        6 : "やや活発で、エネルギーが感じられます。",
        7 : "活発でエネルギッシュな様子が伝わります。",
        8 : "非常に高いテンションで、活動的です。",
        9 : "極めてエネルギッシュで、興奮しています。"
    }
    
    mentality = df['normalized_scores'].mean()
    tension = ActiveSentiment(df['sentiment'])
    
    st.write("全体のポジティブ度：" + to_percentage(mentality))
    st.info(mentality_descriptions.get(int(np.floor((mentality*10,0))[0])))
    st.write("全体のテンション：" + to_percentage(tension))
    st.info(tension_descriptions.get(int(np.floor((tension*10,0))[0])))
    
    # 感情折れ線チャートを表示
    st.plotly_chart(fig)
    
    # Streamlitでヒストグラムを表示
    st.markdown('**文章ごとの感情スコア**')
    row_number = st.slider(
        "文章ごとの感情スコア", 
        min_value=1, 
        max_value=len(emo_df), 
        value=1,
        format="%d行目",
        label_visibility='hidden'
        )
    row_number = row_number-1
    st.write(df['sentence'][row_number])

    selected_row = emo_df.T[row_number]
    barfig = px.bar(
        selected_row,
        labels=dict(index=df['sentence'][row_number],value='感情スコア'),
        range_y=[0.0,1.0],
        color=[6,5,4,3,2,1],
        color_continuous_scale='viridis'
        )
    st.plotly_chart(barfig)
    
    #  カラムの比率を指定
    col1, col2 = st.columns((2,1))
    
    with col1:
        # 列ごとの平均値を計算
        emo_df_mean = emo_df.mean().reset_index()
        emo_df_mean.columns = ['感情', 'スコア']
        
        # 円グラフを作成
        piefig = px.pie(emo_df_mean, names='感情', values='スコア',title='感情の平均スコアの割合',
                        color='感情',
                        color_discrete_map={
                            "喜び": "#fde725",
                            "期待": "#b5de2b", 
                            "驚き": "#35b779", 
                            "恐れ": "#26828e", 
                            "悲しみ": "#3e4989", 
                            "怒り": "#440154"},
                        width=400
                        )
        st.plotly_chart(piefig)
    
    with col2:
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        # max_labelの各値の出現回数を計算
        label_counts = df['max_label'].value_counts()
        # 最頻値を取得
        mode_labels = df['max_label'].mode()
        
        mode_df = pd.DataFrame({
            '最大感情': [df.loc[df['max_score'] == df['max_score'].max(), 'max_label'].iloc[0]],
            '最大スコア':[to_percentage(df['max_score'].max())],
            '最頻値': ['、'.join(mode_labels.tolist())],
            '最頻値の登場数':label_counts.iloc[0],
        })
        st.dataframe(mode_df.T)
    
    
    st.subheader('分析データをダウンロード')
    st.markdown('**データ詳細**')
    st.dataframe(st_df.applymap(to_percentage))
    
    # ダウンロードボタンを作成
    st.download_button(
        label="CSV形式ファイルをダウンロード",
        data=to_write.getvalue(),
        file_name='emotion_data.csv',
        mime='text/csv',
    )