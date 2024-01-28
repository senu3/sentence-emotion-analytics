from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import re

# FastAPIアプリケーションの初期化
app = FastAPI()

# リクエスト用のモデル
class TextAnalysisRequest(BaseModel):
    text: str

# SpacyとTransformersのセットアップ
nlp = spacy.load('ja_ginza')
emotion_model = AutoModelForSequenceClassification.from_pretrained("emotion_model")
tokenizer = AutoTokenizer.from_pretrained("analyze_emotion_tokenizer")
sentiment_pipeline = pipeline("text-classification", model="sentiment_model", tokenizer="analyze_emotion_tokenizer")

@app.get('/')
async def index():
    return{
            'sentence': '入力された文章',
            'sentiment': '感情の積極性',
            'max_label': '最も強い感情',
            'max_score': '最も強い感情の強さ',
            'scores': '全感情の強さ'
        }

# テキスト分析のエンドポイント
@app.post("/analyze")
def analyze_text(request: TextAnalysisRequest):
    text = re.sub(r"\s+", "", request.text)
    doc = nlp(text)

    # モデルを推論モードに
    model = emotion_model
    model.eval()

    # 関数の事前設定

    emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']

    # https://www.delftstack.com/ja/howto/numpy/numpy-softmax/
    def np_softmax(x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    def analyze_emotion(text, show_fig=False, ret_prob=False):
        # 入力データ変換 + 推論
        tokens = tokenizer(text, truncation=True, return_tensors="pt")
        tokens.to(model.device)
        preds = model(**tokens)
        prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
        out_dict = {n: float(p) for n, p in zip(emotion_names_jp, prob)}
        return out_dict

    # sentence, max_label, max_score のリストを出力するコード

    def max_emotion(pred_scores):
        max_label = max(pred_scores, key=lambda label: pred_scores[label])
        max_score = pred_scores[max_label]
        return max_label, max_score

    emotion_list = []
    for sents in doc.sents:
        
        scores = analyze_emotion(sents.text)
        max_label,max_score = max_emotion(scores)
        
        emotion_list.append({
            'sentence': sents.text,
            'max_label': max_label,
            'max_score': max_score,
            'scores': scores
        })

    sentiment_list = []
    for sents in doc.sents:
        scores = sentiment_pipeline(sents.text)
        
        label = int(scores[0]['label'].replace('LABEL_', ''))
        score = scores[0]['score']
        
        sentiment_list.append({
            'sentence': sents.text,
            'sentiment': label
        })
    
    # sentiment_listを辞書に変換
    dict1 = {item['sentence']: item for item in sentiment_list}

    # emotion_listのデータをdict1にマージ
    for item in emotion_list:
        sentence = item['sentence']
        if sentence in dict1:
            dict1[sentence].update(item)
        else:
            dict1[sentence] = item

    # 結果をリストに変換
    return_list = list(dict1.values())

    return return_list