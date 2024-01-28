import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

# Streamlitのタイトル
st.title('テキスト感情分析')

# ユーザー入力用のテキストエリア
text = st.text_area('テキストを入力してください', 
'サンプルテキストです。\n\
雨が降って気分がどんより。\n\
こんな時こそ笑顔が一番！\n\
空が晴れたら虹もかかった。\n\
素晴らしい一日になりました。'
)

# 分析開始ボタン
if st.button('分析開始'):
    # FastAPIサーバーへのリクエスト
    response = requests.post("http://localhost:8000/analyze", json={"text": text})
    
    if response.status_code == 200:
        pred_scores = response.json()
        
        df = pd.DataFrame(pred_scores)
        emo_df = df.loc[:,['sentence','scores']]
        
        # 感情カテゴリ
        positive = ['喜び', '期待', '信頼']
        negative = ['悲しみ', '恐れ', '怒り', '嫌悪']
        natural = ['驚き']

        # 感情リストを展開
        for index, row in emo_df.iterrows():
            for emotion, score in row['scores'].items():
                emo_df.at[index, emotion] = score

        emo_df = emo_df.drop(labels=['sentence','scores'],axis=1)

        # Streamlitでデータフレーム表示
        df['sentiment'] = df['sentiment']-2
        df = df.drop(labels='scores',axis=1)
        st.dataframe(df)

        # ポジティブかネガティブかで重さを計算する関数
        def calculate_sentiment_score(row):
            score = 0
            for emotion in positive:
                score += row.get(emotion, 0)
            for emotion in negative:
                score -= row.get(emotion, 0)
            return score
        
        # カラーマッピング用にノーマライズ
        df['weighted_score'] = emo_df.apply(calculate_sentiment_score, axis=1)
        normalized_scores = (df['weighted_score'] - df['weighted_score'].min()) / (df['weighted_score'].max() - df['weighted_score'].min())

        # データフレーム 'df' から必要なデータを抽出
        sentence = df['sentence'].str[:15]
        sentiment = df['sentiment']

        # 散布図と折れ線チャート
        plt.figure(figsize=(10, 6))

        # カラーマップオブジェクトを作成
        cmap = plt.cm.viridis

        # 線の色を設定するための正規化オブジェクト
        norm = plt.Normalize(vmin=min(normalized_scores), vmax=max(normalized_scores))

        # 散布図のポイントをプロット
        sc = plt.scatter(sentence, sentiment, c=normalized_scores, cmap=cmap, s=df['max_score']*200, alpha=1.0,zorder=2)

        # 各セグメントに色を適用しながら線を描画
        for i in range(len(sentence)-1):
            plt.plot(sentence[i:i+2], sentiment[i:i+2], 
                    c=cmap(norm(normalized_scores[i+1])),
                    alpha=0.7,zorder=1)

        # カラーバー凡例を作成
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm,ax=plt.gca())

        # ラベル
        plt.xticks(rotation=-90)
        plt.xlabel('文')
        plt.ylabel('感情スコア')
        plt.ylim(-3,3)
        plt.title('文ごとの感情スコア')
        
        st.pyplot(plt)
        
        
        # ヒートマップを作成
        plt.figure(figsize=(12, 6))
        plt.title('Emotion Analysis Heatmap')
        plt.ylabel('Sentences')
        plt.xlabel('Emotions')
        plot = sns.heatmap(emo_df, annot=True, cmap="YlGnBu", yticklabels=df['sentence'].str[:10], xticklabels=emo_df.columns)
        st.pyplot(plot.get_figure())
    else:
        st.error("エラーが発生しました。")