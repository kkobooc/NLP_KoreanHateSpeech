한국어 악플 분류를 위한 머신러닝 프로젝트
========================================
본 프로젝트는 Kaggle Competition에서 사용된 데이터 셋을 이용하여 진행되었습니다. 
* [Korean hate speech (Kaggle)](https://www.kaggle.com/c/korean-hate-speech-detection/overview)

#### **프로젝트 결과: 프로젝트 종료일 2021.01.28. Leaderboard 8위 기록(Score: 0.57696)**
![image](https://user-images.githubusercontent.com/67237187/112584918-267f7000-8e3c-11eb-922f-e9bb71c06d68.png)

I. 프로젝트 개요
----------------
1. 프로젝트 배경 및 목적
#### 댓글을 통해서 긍정적인 힘을 얻을 수도 있지만, 악성 댓글로 인해 사람을 망치는 경우가 더 많습니다. 연예인들을 향한 악성 댓글이 특히 더 큰 문제가 되어 왔으며, 이를 해결하기 위해 실명제를 적용해야 한다는 의견들에 결국 연예기사에는 댓글을 작성할 수 없게 되었습니다. 다음과 네이버는 결국 연예기사에 댓글을 작성할 수 없게 만들었고 이는 근본적인 해결 방법이 되지 못합니다.
#### 본 머신러닝 프로젝트에 사용된 데이터셋을 구성한 사람의 의도처럼, 다양한 모델을 사용하며 혐오 발언을 구분할 수 있는 최적의 방법을 찾아보기로 하였습니다.
  
2. 데이터셋 소개
 ### Train data와 Validation data에는 댓글 내용의 수위에 따라 구분되어 있습니다:
  - hate: 글의 대상 또는 관련 인물, 글의 작성자 또는 댓글 등에 대한 강한 혐오 또는 모욕이 표시 됨
  - offensive: 비록 논평이 hate만큼 혐오스럽거나 모욕스럽지 않지만 대상이나 독자를 불쾌하게 만듦
  - none: 어떠한 증오나 모욕도 내포되지 않음
  ##### Train data: 7896 rows / Validation data: 471 rows / Test data: 974 rows (no label)
  
3. EDA
- 2글자 이상 명사 통계: none 라벨과 비교했을 때, offensive와 hate 댓글에서 성별, 비속어, 혐오 표현 단어의 빈도가 높음

![image](https://user-images.githubusercontent.com/28764376/105451950-cac02b80-5cc0-11eb-8c29-5ce707c3d863.png)
![image](https://user-images.githubusercontent.com/28764376/105451997-dd3a6500-5cc0-11eb-96d0-503337a531ea.png)

- 형태소 기준 2-gram 통계: 형태소로 분석해 보았을 때도 마찬가지로 hate 댓글에서 성별, 비속어, 혐오 표현 단어의 빈도가 높음

![image](https://user-images.githubusercontent.com/28764376/105451226-82543e00-5cbf-11eb-8fcd-601ba1c148a7.png)
![image](https://user-images.githubusercontent.com/28764376/105451256-9009c380-5cbf-11eb-923d-9bebc9b79bfc.png)

- 형태소 기준 2-gram 통계: 부정적인 댓글일수록 토큰 수가 많고, 문장의 길이가 더 김

![image](https://user-images.githubusercontent.com/28764376/105451147-55a02680-5cbf-11eb-9f58-57d680e40851.png)

4. 설치 패키지 
- Python 3 
```
pip install pykospacing
pip install kss
pip install konlpy
pip install soynlp
git clone https://github.com/kakao/khaiii.git # khaii 설치 방법 참조
pip install gensim
pip install jamo
pip install transformers
pip install sentencepiece
pip install hanspell
```

II. 전처리
-----------
1. 적용 모델에 따라 전처리를 다르게 진행하거나, 아예 전처리를 하지 않은 부분도 있으나 기본적인 전처리는 아래와 같은 프로세스로 진행 되었음:
- 특수문자 제거(자체 함수 사용)
- 문장 분리(kss)
- 띄어쓰기(pykospacing)
- 중복 제거(soynlp.normalize)

2. 토크나이저
- Khaiii
- Okt
- Mecab

III. 모델별 실험
----------------
1. 벡터화: TF-IDF / Count Vectorizer 적용 및 비교
- 공통 옵션값 적용후  f1-score 비교
  * Count Vectorizer 옵션값  : min_df=0.0, analyzer='char', ngram_range=(1,3), max_features=5000
  * TF-IDF Vectorizer 옵션값 : min_df=0.0, analyzer='char', ngram_range=(1,3), max_features=5000, sublinear_tf=True
  * TF-IDF Vectorizer의 f1-score가 조금 우세한것을 확인할수 있음
  * [벡터라이징 기법 비교](https://github.com/hayoon/nlp_hate_speech/blob/master/code/gijoong/02_cv_tfidf_compare.ipynb)
```
KFold f1_score  :: TF :  0.5519975598739248 , CV :  0.5395692760168908
SKFold f1_score :: TF :  0.5531487999460587 , CV :  0.5410395492065031
dev 파일        :: TF :  0.5773105429455988 , CV :  0.5594804815636172
```

2. 머신러닝 모델: 직접 함수를 생성하기도 하고 다양한 모델링 기법을 사용하며 성능을 개선하기 위해 비교해 보았음
- Naive Bayes
  * 두 가지 전처리 방법으로 모델링을 시도하여, 특수문자와 공백을 모두 제거한 뒤 띄어쓰기한 경우에 성능이 더 좋았음
  * 나이브 베이즈의 'Most informative features'를 통하여 각 라벨에 영향을 주는 단어들을 확인할 수 있었음
  * validation data: F1-score 0.525420
  * [나이브 베이즈를 이용한 분류 예측](https://github.com/hayoon/nlp_hate_speech/blob/master/code/yeji/02_naivebayes_trial.ipynb)
- 코사인 유사도-1
  * 훈련 데이터를 각 레이블별로 분류해 TFIDF Vectorize한 뒤, 각각의 평균 벡터값 산출
  * 'comment' 컬럼의 벡터값 <-> 각 레이블 평균 벡터값 간의 유사도 측정 후, 가장 유사도가 높은 레이블 리턴
  * validation data: F1-score 0.498306
  * [코사인 유사도를 이용하여 만든 분류 모델](https://github.com/hayoon/nlp_hate_speech/blob/master/code/jc/05_Cosine_Similarity.ipynb)
- 코사인 유사도-2
  * 댓글을 입력할 시, 코사인 유사도 top3 댓글을 출력하는 함수를 생성. khaiii를 이용한 형태소 분석 이전의 Naive Bayes를 써서 얻은 informative feature의 상위 100개에 포함되는 품사 단어들만 vocabulary list에 추가하여 분석하도록 함
  * 출력된 유사도가 높은 댓글 top3의 라벨이 모두 일치할 경우, 해당 라벨과 가장 유사하다고 판단하여 같은 라벨로 예측
  * 출력된 유사도가 높은 댓글 top3의 라벨이 두 가지가 나왔을 경우, 유사도 값의 차이가 있을 수 있기 때문에 같은 라벨을 가진 값들끼리 유사도를 더하며, bias가 존재한다면 0.1을, 이것이 gender bias라면 0.2를 또 더해줘서 최종적으로 높은 값을 가지는 라벨로 예측
  * 출력된 유사도가 높은 댓글 top3릐 라벨이 모두 다르다면, 부정적인 댓글이 최수 두 가지 (offensive, hate)이기 때문에 마찬가지로 bias에 대한 가중치를 적용하고, 이 두 라벨 중 유사도가 더 높은 라벨로 예측
  * validation data: F1-score 0.448089
  * [코사인 유사도를 이용하여 만든 분류 모델2](https://github.com/hayoon/nlp_hate_speech/blob/master/code/hayoon/cos_sim_predict_label.py)
- 그 외 전처리하지 않고 기본 파라미터로 모델간 비교 실험(Test: Validation data)
  * [다양한 모델 이용한 분류 예측](https://github.com/hayoon/nlp_hate_speech/blob/master/code/jc/02_2_Model_Comparison.ipynb)
  * 전반적으로 Logistic Regression이 우수한 성능을 보여, Logistc Reg. 중심으로 성능 개선 시도
```
  Model : RandomForestClassifier()
         F1 Score  Accuracy
  Train  0.998677  0.998734
  Test   0.487739  0.513800
  ------------------------------
  Model : LogisticRegression()
         F1 Score  Accuracy
  Train  0.832283  0.841565
  Test   0.579778  0.585987
  ------------------------------
  Model : SVC()
         F1 Score  Accuracy
  Train  0.973995  0.974924
  Test   0.536260  0.552017
  ------------------------------
  Model : LGBMClassifier()
         F1 Score  Accuracy
  Train  0.787965  0.796479
  Test   0.546274  0.552017
  ------------------------------
  ```
  
IV. Logistic Regression에 집중한 분류
--------------------------------------
1. JAMO 토크나이저 사용
- 온라인 댓글 특성상 맞춤법이 지켜지지 않아 형태소별 토큰화의 정확도가 떨어진다고 판단, 자/모 분리하여 토큰화 시도
- ngram(1, 6)에서 가장 우수한 성능을 보임
- Validation Data 예측 F1-Score: 0.619 / Kaggle score : 0.516
* [JAMO 토크나이저 사용한 Logistic Regression](https://github.com/hayoon/nlp_hate_speech/blob/master/code/jc/04_JAMO.ipynb)
2. Word2Vec 사용
- 기존 Count/TFIDF Vectorizer는 단어의 의미와 문맥을 파악하지 못하는 단점이 존재
- 추가로 제공된 댓글 데이터 중 100만개를 랜덤 샘플링하여 Word2Vec 모델 학습
- 학습된 Word2Vec을 바탕으로 각 문장들을 Vectorize하여 Logistc Regression으로 예측
- 모델 테스트시 유사 단어 추출은 좋았으나 실질적 악플 분류에 뛰어난 성능을 보여주지는 못함
- Validation Data 예측 F1-Score: 0.546 / Kaggle score : 0.47134
* [Word2Vec 사용한 Logistic Regression](https://github.com/kkobooc/NLP_KoreanHateSpeech/blob/master/code/Jaecheol/06_Word2Vec.ipynb)
* [Word2Vec model](https://github.com/kkobooc/NLP_KoreanHateSpeech/blob/master/code/Jaecheol/06_01_comments.model)
3. Doc2Vec
- validation data와 train data를 합쳐서 train과 test로 split하여 학습 시킴
- 각 코멘트 뒤에 라벨을 태그로 입력하여 dbow와 dm 두 가지 방법으로 벡터화한 후 각각 모델링한 값과 둘을 concat하여 모델링한 값 3가지를
- F1-score: 0.546942 / Kaggle score: 0.49027
* [Doc2Vec 사용한 Logistic Regression](https://github.com/hayoon/nlp_hate_speech/blob/master/code/yeji/09_doc2vec.ipynb)
4. 시도
- 전처리 없이 - TF-IDF Vectorizer - Logistic Regression
  * [tf-idf 옵션값 테스트](https://github.com/hayoon/nlp_hate_speech/blob/master/code/gijoong/03_tf_idf_option_test.ipynb)
  * Validation Data 예측 F1-Score: 0.619 / Kaggle score : 0.528
- 전처리 - TF-IDF Vectorizer - Logistic Regression
  * 최고점 갱신 옵션값
    * 전처리 : repeat_normalize - maxscore_tokenizer
    * TF-IDF : min_df=0.0, analyzer='char', ngram_range=(1,3), sublinear_tf=True, max_features=100000
  * [전처리 진행 방법](https://github.com/hayoon/nlp_hate_speech/blob/master/code/gijoong/04_preprocessing.ipynb)
  * Validation Data 예측 F1-Score: 0.625 / Kaggle score : 0.532

V. 딥러닝 (KoBert)
----------------
 - https://github.com/monologg/KoBERT-Transformers 에서 지원하는 pretrained tokenizer 사용
 - config :
 ```
  "max_seq_len": 128,
  "num_train_epochs": 10,
  "adam_epsilon": 1e-8,
  "train_batch_size": 32,
  "eval_batch_size": 64,
  "learning_rate": 5e-5
 ```
 
 - [KoBert를 이용한 분류](https://github.com/hayoon/nlp_hate_speech/blob/master/code/hayoon/kobert_multiclass.ipynb)
 - Validation Data 예측 F1-Score: 0.637 / Kaggle score : 0.57696
 
 - Sample outputs on unlabeled test data
 
 ![image](https://github.com/hayoon/nlp_hate_speech/blob/master/data/sample_output_01.png?raw=true)
 ![image](https://github.com/hayoon/nlp_hate_speech/blob/master/data/sample_output_02.png?raw=true)
 ![image](https://github.com/hayoon/nlp_hate_speech/blob/master/data/sample_output_03.png?raw=true)

Built with: 
----------
- 김예지
  * 전처리, 다양한 머신러닝 모델링, 코사인 유사도 이용한 분류, doc2vec 사용하여 모델 성능 개선 시도
  * Github: https://github.com/yeji0701
- 이기중
  * 전처리, 다양한 머신러닝 모델링, Logistic Regression에 집중한 성능 개선
  * Github: https://github.com/GIGI123422
- 정하윤
  * 전처리, 다양한 머신러닝 모델링, 코사인 유사도 이용한 분류, KoBERT 이용하여 성능 개선
  * Github: https://github.com/hayoon
- 최재철
  * 전처리, 토크나이저별/모델별 성능 비교, 코사인 유사도 이용한 분류, JAMO 토크나이저 사용, Word2Vec을 이용한 모델 성능 개선 시도
  * Github: https://github.com/kkobooc
  
 Acknowledgements:
 -----------------
 https://github.com/inmoonlight/koco
