---
title: Transformer 모델
date: 2024-08-23
categories: ["2024", "LLM"]
tags: ["transformer", "attention"]
use_math: true
---

## 기존 RNN 구조

<img width="739" alt="Screenshot 2024-08-23 at 3 23 18 PM" src="https://github.com/user-attachments/assets/4877319c-e7a2-46d9-942a-a20b83ad3001">
<span style="color:gray; font-size: 80%">By fdeloche - 자작, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=60109157</span>


* RNN (Recurrent Neural Network)는 순차적. 병렬 처리 불가능
* $ h_t = f(h_{t-1}, x_t) $
    * 이전 단계에 의존한다.
    * 긴 sequence 처리할 때 계산 시간 증가.
* 메모리 제약으로 배치 크기를 줄여야 할 수도.


* 입력 길이가 길어지면 먼저 입력한 토큰 정보가 희석됨 → performance 하락
* 층을 깊이 쌓으면 gradient vanishing, gradient exploding → 학습 불안정

----


## Transformer

### 텍스트 임베딩

<img width="785" alt="Screenshot 2024-08-23 at 3 44 18 PM" src="https://github.com/user-attachments/assets/550d55da-f36f-4fe5-b990-9b90f6837b56">


#### 토큰화
1. 텍스트를 파싱하여
2. 각 토큰에게 숫자 ID를 부여하고
3. 어떤 토큰이 어떤 숫자 ID에 해당하는지 제시하는 vocabulary(사전) 생성

* 토큰화 단위가 너무 커지면 OOV (Out Of Vocabulary)  
    토큰화 단위가 너무 작으면 vocabulary 크기 증가.  
    텍스트 의미 유지 X  
    $\therefore$ 적절한 크기로 토큰화
* subword 토큰화 : 자주 나오면 단어는 있는 그대로, 가끔 나오는 단어는 더 작은 단위로 토큰화



#### 토큰 임베딩
<img width="300" alt="Screenshot 2024-08-23 at 3 46 40 PM" src="https://github.com/user-attachments/assets/b2666a18-b9d3-4227-88f5-a4d54847e6ce">
  
토큰 : 아직 의미가 없는 n차원 벡터.  
임베딩 층에 벡터를 다 때려 넣어 → 이 과정에서 임베딩 층이 학습됨 → 토큰의 의미를 잘 담은 임베딩 생성


#### 위치 임베딩
트랜스포머 : 모든 입력 동시 처리 → 순서 정보 소실  
텍스트에서 순서 정보는 중요하기 때문에, 다시 추가를 해준다 (= 위치 인코딩)

* absolute position encoding (절대적)
    * 수식으로 위치 정보 추가
    * 임베딩으로 위치 정보 학습 (모델로 추론)
    * pros) 간단하게 구현 가능
    * cons) 토큰 사이의 상대적 위치 정보 활용 불가능. 긴 텍스트(학습 데이터 없음) 추론 시 성능 저하
* relative position encoding (상대적)
    <img width="1294" alt="Screenshot 2024-08-23 at 4 02 37 PM" src="https://github.com/user-attachments/assets/a3a93424-895c-4411-b517-89b531649369">

---------

### Attention

self attention
: 입력된 문장 내에서 각 단어가 어떤 관련이 있는지 계산 → 각 word의 representation 조정

* 확장성 : 더 깊은 모델 구축 가능. 동일한 블록 재사용.
* 효율성 : 병렬 연산 가능. 처리 시간 감소.
* 더 긴 입력 처리에서 성능 저하 없음

1. 단어와 단어 사이의 관계 계산
    → 관련이 깊은 단어와 낮은 단어 구분
2. 관련이 깊은 단어는 더 많이 맥락 반영
    관련이 적은 단어는 더 적게 맥락 반영

벡터 내적 → 유사도
    주변 맥락 반영, 유사한 의미의 키 검색
<img width="654" alt="Screenshot 2024-08-23 at 4 11 12 PM" src="https://github.com/user-attachments/assets/e4ee6b6d-f0aa-4b74-8b93-03841db2543f">

* 각 Query에 대해 모든 key와의 compatibility 계산 → softmax를 통해 가중치로 변환
* 각 Value 벡터에 해당하는 가중치 곱해
* 가중치가 적용된 Value 벡터들을 모두 더해 → $Output = \sum{(weight_i * value_i)}$
    * 당연히 Output도 벡터다.


cons)
1. 같은 단어(동음이의어)끼리 임베딩 동일 → 관련도가 크게 계산되면서 주변 맥락 반영을 잘 못 할 수 있음
2. 간접적인 관련성을 반영하지 못함  
    $\therefore$ 토큰 임베딩을 변환하는 가중치 $W_Q$, $W_K$ 도입  
    → 토큰과 토큰 사이 관계 계산하는 능력 학습시킴




#### Scaled Dot-Product Attention

트랜스포머는 dot-product attention을 사용.  

$$ Attention(Q, K, V)=softmax(\frac{QK^T}{\sqrt{d_k}})V $$
* $softmax(\frac{QK^T}{\sqrt{d_k}})$ : 가중치
* $V$ : 행렬. 각 행이 하나의 value 벡터. (sequence_length x $d_v$) 차원
    * $d_v$ : 벡터의 차원

$1/\sqrt{d_k}$ 로 additive attention의 일부 장점도 얻음.
* $\sqrt{d_k}$ : scaling. 분산을 1로 정규화.  
    분산 → 표준편차 변환과 비슷하다고 생각하면 될 듯.



#### Multi-Head Attention
: 여러 어텐션 연산 동시 적용 (헤드 수 만큼)

1. 쿼리, 키, 값을 헤드 수로 쪼개고,
2. 각각 어텐션을 계산한 다음 입력과 같은 형태로 다시 변환.
3. 선형 층을 통화시켜서 최종 결과 반환.

각 헤드에서의 Q, K, V는 $d_{model}/h$ 차원.  
줄어든 차원에서 attention 연산.  
결과 다 concatenate. 연결된 거 다시 $d_{model}$ 차원으로 변환.  

입력 x <span style="color:red">$W_Q$</span> = $Q$  
입력 x <span style="color:red">$W_K$</span> = $K$  
입력 x <span style="color:red">$W_V$</span> = $V$  
    * 학습 가능한 가중치 행렬  
    * 각 헤드는 서로 다른 가중치 행렬 사용 중
  

Encoder-Decoder Attention :
* 디코더의 각 층에서 발생.
* 각 디코더 층에서는 인코더의 최종 출력을 사용.
    * 인코더의 최종 출력 : 인코더 전체 (6개 층)를 통과한 것.  
크로스 어텐션 : 인코더의 결과를 디코더가 활용



------

### Encoder Decoder
<img width="726" alt="Screenshot 2024-08-23 at 4 26 25 PM" src="https://github.com/user-attachments/assets/22daec6a-1212-4eb0-ac74-73a87295bd6d">  
<span style="color:gray; font-size: 80%">출처: 위키독스 https://wikidocs.net/162096</span>


인코더 스택 : 6개의 동일한 인코더 층으로 구성. 각 층은 독립적으로 작동  
디코더 스택 : 6개의 동일한 디코더 층으로 구성. 각 층은 독립적으로 작동

* 각 층은 Multi-Head Attention 과 Feed Forward 2개의 sub-layer를 가짐
* 각 sub-layer 이후 layer-normalization을 함.
    * 잔차 연결 후에 각 layer의 출력을 정규화

인코딩 : 입력 시퀀스가 인코더 스택을 한 번 통과.  
- 6개의 인코더 층은 순차적으로 모두 통과.
- 각 층의 출력이 다음 층의 입력으로 사용됨.  
- 각 층 내에 multi-head attention. (논문에서는 n_head = 8)  
- 각 층 내에서는 self-attention 병렬 연산 처리 가능.  
- 각 층 거치면서 입력의 표현이 점점 더 추상화되고, 고차원적인 특징을 포착함.  
- 각 층에서 잔차 연결.  

디코딩도 동일

### Residual Connection
: 각 sub-layer의 입력을 출력에 더해.

* 원본 정보가 직접적으로 다음 layer로 전달. 학습 향상.
* sub-layer는 원본 정보에 대한 “잔차”(=”변화”)를 학습하게 됨.
* 중요한 변환을 학습하지 못해도 최소한 원본 정보는 보존됨
* vanishing gradient 문제 완화 (역전파 시 그래디언트가 residual connection을 통해 직접적으로 이전 layer들로 전달 될 수 있음)



------

### Position-wise Feed-Forward Networks
: 데이터의 특징을 학습하는 fully connected layer. 입력 텍스트 전체를 이해하는 역할  
두 개의 선형 변환과 ReLU 활성화 함수 사용.  

* 특징 추출. 불필요한 정보 제거
* 차원 변환. 입력을 고차원 공간으로 투영
* sparse → 계산량 감소
* 일부 정보(음수) 소실 → 더 중요한 특징에 집중

$$ FFN(x)=max(0, xW_1 + b_1)W_2 + b_2 $$

1. 첫 번째 1x1 convolution  
    $xW_1 + b_1$
2. ReLU 활성화
3. 두 번째 1x1 convolution  
    (결과)$W_2 + b_2$

cf)  
ReLU (Rectified Linear Unit)
- 활성화 함수 $f(x) = max(0, x)$  
- feed forward 네트워크에 비선형성 도입  
    → 모델이 복잡한 패턴 학습할 수 있도록

--------

### 학습

인코더 : 멀티 헤드 어텐션  
디코더 : 마스크 멀티 헤드 어텐션

디코더의 본질은 생성.  
앞에서 생성한 토큰을 기반으로 다음 토큰 생성.  
순차적. causal(인과적), auto-regressive(자기 회귀적)  

실제 텍스트 생성 시에는 지금까지 생성된 텍스트만 확인 가능.  
but 학습할 때는 뒤의 내용이 다 보여  
따라서, 그 이전에 생성된 토큰까지만 확인할 수 있도록 마스크 추가  
<img width="326" alt="Screenshot 2024-08-23 at 3 17 40 PM" src="https://github.com/user-attachments/assets/89d88e2f-a972-4f87-bc12-35a654636ac4" style="zoom:50%;">  
이런 거 곱하면 -inf 곱해지는 부분의 값이 0이 됨  



-------

## 출처

“Attention Is All You Need” arXiv:1706.03762 [cs.CL]  
<LLM을 활용한 실전 AI 애플리케이션 개발> https://github.com/onlybooks/llm
