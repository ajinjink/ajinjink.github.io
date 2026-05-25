---
title: "[ML] Overfitting & Regularization"
date: 2026-05-24
categories: ["AI", "ML"]
tags: ["training loss", "validation loss", "overfitting", "model capacity", "early stopping", "regularization", "shrinkage method", "ridge regression", "RSS", "penalty", "bias-variance tradeoff", "lasso", "cross validation", "model selection"]
use_math: true
---


## Model Selection & Cross-validation

모델을 학습시킬 때 데이터를 세 partition으로 나눠서 사용함.

- **Training set (70%)**: 모델의 파라미터를 학습하는 데 사용
- **Validation set (20%)**: 하이퍼파라미터 선택 (model selection)에 사용
- **Eval set (10%)**: 최종 모델의 성능을 평가하는 데 사용

예를 들어 세 가지 후보 모델이 있다고 하자.

| Model   | Train error | Val error | Final error |
| ------- | ----------- | --------- | ----------- |
| Model 1 | 2.5%        | 3.6%      | -           |
| Model 2 | 2.7%        | 3.2%      | **3.1%**    |
| Model 3 | 2.6%        | 3.4%      | -           |

Val error가 가장 낮은 Model 2를 고른 다음, eval set에서 최종 성능(3.1%)을 측정함.

그런데 여기서 한 가지 의문이 생김: **이 error들을 학습 과정 중 언제 측정해야 하는가?**

---

## Training and Validation Curves

<img width="80%" alt="validation curve" src="/assets/images/2026-05-25/validation-curve.png">

실제 학습 과정을 관찰해보면

- **Training loss**는 학습 내내 꾸준히 감소함
- **Validation loss**는 어느 시점에서 최저점을 찍은 뒤, 다시 증가하기 시작함

학습 중 일어나는 일을 세 단계로 나눠보면:

1. **초기 단계**: training loss와 validation loss 둘 다 빠르게 감소
2. **중간 단계**: validation loss가 평탄해지는 반면, training loss는 계속 감소
3. **후반 단계**: validation loss가 **오히려 증가**하기 시작하는데 training loss는 여전히 감소

왜 이런 일이 일어날까?

---

## Overfitting

우리의 진짜 목표는 학습 데이터가 아니라 **unseen test examples**에서 잘 작동하는 모델을 만드는 것임. 

그렇다면 왜 어느 시점 이후로 eval loss가 더 나빠지는가?

<img width="60%" alt="eval loss" src="/assets/images/2026-05-25/eval-loss.png">

- **초반**: 모델이 데이터에서 가장 중요한 일반적인 패턴을 학습함. 이 패턴은 다른 데이터에도 generally applicable
- **후반**: 이 시점 이후로 모델이 학습하는 것은 training set에만 우연히 존재하는 **노이즈**

- 이렇게 **training loss는 좋아지는데 eval loss는 오히려 나빠지는 현상**을 **overfitting**이라 부름.

### Overfitting의 특징

- 모델이 일반적인 trend뿐만 아니라 **training data의 노이즈까지** 학습함
- Overfitting 이후로 training set 성능은 계속 좋아지지만, eval set 성능은 나빠짐
- 특히 **high-dimensional data**나 **noisy features**를 다룰 때 주의해야 함

### Overfitting과 Model Capacity

Overfitting은 사실 **model capacity**와 깊은 관련이 있음.

- 모델이 **불필요하게 복잡**하면 overfitting되기 쉬움 → 노이즈까지 memorize할 수 있는 capacity가 있기 때문
- 반대로 모델 capacity가 **불충분**하면 일반적인 패턴조차 학습 못함 → **underfitting**

세 가지 예시를 보면:
<img width="60%" alt="model dimensions" src="/assets/images/2026-05-25/model-dimensions.png">

- $g(\beta_0 + \beta_1 x_1 + \beta_2 x_2)$ → 너무 단순 (underfit, 직선 decision boundary)
- $g(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2)$ → 적절함 (부드러운 곡선)
- $g(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2 + \beta_6 x_1^2 x_2 + \beta_7 x_1 x_2^2 + \beta_8 x_1^2 x_2^2 + \beta_9 x_1^3 + \ldots)$ → 너무 복잡 (overfit, 구불구불한 boundary)

결국 두 가지 질문이 남음:

1. **언제 training을 멈춰야 하는가?**
2. **모델의 right capacity는 무엇이고 어떻게 결정하는가?**

---

## Early Stopping

### 언제 학습을 멈춰야 할까

- 데이터가 **i.i.d. assumption** 하에 무작위로 나뉘었다면, training/validation/eval 세 partition 모두 비슷한 probability distribution을 따름. 

- 각 set이 극단적으로 작지 않은 이상, **세 partition 모두 비슷한 시점에 overfit하기 시작**한다고 가정할 수 있음.
  - 일반적인 패턴은 모든 partition에 applicable하지만, training partition 특유의 노이즈는 validation set에서도 여전히 노이즈일 뿐.

- 따라서 **set-aside validation set으로 loss를 추적**하다가 overfit이 시작될 때 학습을 멈추는 방법을 **Early Stopping**이라 부름.

### 어떤 metric으로 overfit을 판단할까

training에 쓰인 loss function으로 판단해야 하는가, 아니면 우리가 진짜 관심 있는 metric으로 판단해도 되는가?

- 예를 들어 logistic regression은 **log-loss**를 최소화하지만, 우리가 실제로 관심 있는 건 보통 **classification accuracy**임.

- 내가 사용하는 loss function이 계산이 안 돼서(e.g. 미분 불가/계산량 초과) 대체하는 다른 loss를 사용하는 경우, 실제 결과와 다를 수 있음

  - Observation 1: 어떤 경우에는 surrogate loss는 overfit하지 않는데, 실제 metric(Average Precision, NDCG@10)은 overfit함.
    - 예: Log[A] loss는 validation loss가 안정적인데 Avg Prec와 NDCG는 계속 나빠짐 
    - → **validation loss에만 의존하면 안 됨**

  - Observation 2: 각 metric은 서로 다른 시점에 overfit함. 
    - 예: Exp[M]은 다른 세 loss와 달리 크게 overfit하지 않음 
    - → **가장 중요한 metric을 early stopping 기준으로 사용**하는 게 좋음




----

## Regularization

### Model Selection: 다른 접근

두 번째 질문 "모델의 right capacity를 어떻게 결정할까"로 넘어감.

- 항의 개수가 많아지면 = 파라미터 $\beta_j$가 많아지면 = 더 복잡한 모델 = 더 높은 capacity
- 항이 많을수록 training data의 fine detail을 fit할 자유도가 커짐

직관적인 방법은 capacity가 다른 여러 후보 모델을 시도해보고 validation error가 가장 낮은 것을 고르는 것임.



근데 더 좋은 방법은 없을까?

- **더 단순한 모델은 더 복잡한 모델의 special case**임.

  - 2차 모델은 $\beta_3, \beta_4, \beta_5 = 0$이면 1차 모델과 같음
    - $g(\beta_0 + \beta_1 x_1 + \beta_2 x_2)$
    - $g(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2)$

  - 고차 모델도 $\beta_6, \beta_7, \beta_8, \beta_9 = 0$이면 저차 모델과 같음
    - $g(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2 + \beta_6 x_1^2 x_2 + \beta_7 x_1 x_2^2 + \beta_8 x_1^2 x_2^2 + \beta_9 x_1^3 + \ldots)$ 

- **가장 복잡한 모델에서 시작해서, 정말 필요할 때만 $\beta_k$가 non-zero가 되도록 유도**하는 방법을 쓰면 됨.

---

### Shrinkage Methods

모든 $p$개의 predictor를 다 포함하는 모델을 fit하되, coefficient estimate을 **제약(constrain)** 또는 **정규화(regularize)** 하는 기법. 
- coefficient를 0 쪽으로 **shrink**시킴.
- **coefficient를 shrink하면 그들의 variance를 크게 줄일 수 있음**.


---

### Ridge Regression

Least square 방법은 다음을 최소화하는 $\beta_0, \beta_1, \ldots, \beta_p$를 추정함:

$$ \text{RSS} = \sum_{i=1}^{n} (\textbf{y}_i - (\hat{\beta}_0 + \hat{\beta}_1 \textbf{x}_i))^2 $$

**Ridge regression**은 $\beta$ 값이 커지는 것에 페널티를 주는 항을 추가함:

$$ \text{RSS} = \sum_{i=1}^{n} (\textbf{y}_i - (\hat{\beta}_0 + \hat{\beta}_1 \textbf{x}_i))^2 + \lambda \|\beta\|_2^2 $$

- $\lambda \geq 0$은 regularization 강도를 조절하는 하이퍼파라미터
- $\beta = [\beta_1, \ldots, \beta_p]^\top \in \mathbb{R}^p$
- $\|\|\beta\|\|_2^2 = \beta_1^2 + \cdots + \beta_p^2$
- $\|\|\beta\|\|_2^2$가 커지면 cost가 커지는 것.
  - 앞의 모델을 fit 하는 것과 모델 차원을 줄이고자 하는 항 사이의 trade off

Ridge regression도 least squares와 마찬가지로 RSS를 작게 만들어 데이터에 잘 fit하려 함. 

- 하지만 추가된 **shrinkage penalty** $\lambda \|\beta\|_2^2$는 $\beta_1, \ldots, \beta_p$가 0에 가까울 때 작아지므로, $\beta_j$를 0 쪽으로 끌어당기는 효과가 있음.

- $\lambda$는 두 항의 상대적 영향력을 결정하는 tuning parameter임. 

- **좋은 $\lambda$를 선택하는 것이 매우 중요**하며, cross-validation을 사용함.

  - $\lambda \to \infty$: 페널티가 너무 커서 모든 $\beta_j$가 0으로 강제됨

  - $\lambda = 0$: 페널티가 없어져서 unregularized linear regression과 동일

---

#### Predictor의 Scaling

표준 least squares coefficient estimate은 **scale equivariant** 함. 

- $x_j$를 상수 $c$로 곱하면 coefficient는 $1/c$로 scale됨. 

- 따라서 $\beta_j x_j$는 그대로 유지됨.

$\leftrightarrow$ ridge regression의 coefficient estimate은 predictor의 scale에 따라 **substantially change** 함. 

- 이는 penalty 항이 coefficient의 squared sum이기 때문임.

- 따라서 ridge regression을 적용하기 전에 predictor를 **standardize**하는 것이 좋음

- $$ \widetilde{x}_{ij} = \frac{x_{ij}}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \bar{x}_j)^2}} $$

---

#### Ridge Regression이 작동하는 이유

$$ \text{MSE}(\hat{\theta} | \mathbf{X}) = \text{Var}(\hat{\theta} | \mathbf{X}) + \text{Bias}(\hat{\theta} | \mathbf{X})^2 $$

$\lambda$가 커질수록 (= regularization이 강해질수록):

- **Variance는 감소**
- **Bias는 증가**
- $\lambda$가 너무 크면 모델이 너무 단순해져서 Bias가 너무 커져 $\to$ MSE 증가
- $\lambda$가 너무 작으면 오버피팅이 너무 심해져서 Variance가 커져 $\to$ MSE 증가
- 적절한 지점에서 **MSE가 최소**

즉 ridge regression은 **bias-variance tradeoff** 관점에서 약간의 bias를 받아들이고 variance를 크게 낮추는 전략임.

---

#### Matrix Notation으로 유도

Ridge regression의 objective를 matrix notation으로 쓰면:

$$ \hat{\beta} = \underset{\beta}{\arg\min} , (\mathbf{Y} - \mathbf{X}\beta)^\top (\mathbf{Y} - \mathbf{X}\beta) + \lambda \|\beta\|_2^2 $$

MLE를 구하기 위해 $\beta$에 대해 편미분:

$$ \frac{\partial \text{RSS}(\beta)}{\partial \beta} = -\mathbf{X}^\top (\mathbf{Y} - \mathbf{X}\beta) - (\mathbf{Y} - \mathbf{X}\beta)^\top \mathbf{X} + 2\lambda\beta $$

두 번째 항은 첫 번째 항의 transpose이고 scalar이므로:

$$ = -\mathbf{X}^\top (\mathbf{Y} - \mathbf{X}\beta) - \mathbf{X}^\top (\mathbf{Y} - \mathbf{X}\beta) + 2\lambda\beta $$

$$ = -2\mathbf{X}^\top (\mathbf{Y} - \mathbf{X}\beta) + 2\lambda\beta = 0 $$

$$ \mathbf{X}^\top \mathbf{Y} - \mathbf{X}^\top \mathbf{X}\beta - \lambda\beta = 0 $$

$$ \mathbf{X}^\top \mathbf{Y} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}) \beta $$

$$ \boxed{\hat{\beta} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{Y}} $$

- $\hat{\beta} \in \mathbb{R}^{d \times 1}$
- $(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}) \in \mathbb{R}^{d \times d}$
- $\mathbf{X}^\top \mathbf{Y} \in \mathbb{R}^{d \times 1}$

- Unregularized linear regression의 결과 $\hat{\theta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}$와 비교하면, **$\lambda \mathbf{I}$가 추가됐다는 점만 다름**. 

- 추가로 $\lambda \mathbf{I}$ 덕분에 $\mathbf{X}^\top \mathbf{X}$가 singular한 경우(예: $n < p$)에도 역행렬이 존재하게 됨.

---

#### 기하학적 해석

<img width="60%" alt="ridge regression" src="/assets/images/2026-05-25/ridge-regression.png">

Penalty 항 $\lambda \|\|\beta\|\|_2^2$은 $\beta$가 클수록 비용이 커지는 효과를 줌.

- $\beta$가 큰 값을 가지려면 첫 번째 항(RSS)을 줄이는 benefit이 penalty보다 커야 함.

- $\lambda$가 커지면 (예: $\lambda = 0.5$) 허용되는 $\beta$의 norm이 더 작아짐 → 더 강한 정규화
- $\lambda = 0$이면 unregularized linear regression과 동일

---

#### Ridge Regression

| Linear Regression                                            | Ridge Regression                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $\min_\theta (\mathbf{Y} - \mathbf{X}\theta)^\top (\mathbf{Y} - \mathbf{X}\theta)$ | $\min_\theta (\mathbf{Y} - \mathbf{X}\theta)^\top (\mathbf{Y} - \mathbf{X}\theta) + \lambda \|\theta\|_2^2$ |
| $\hat{\theta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}$ | $\hat{\theta} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{Y}$ |

- Objective function에 **penalty term을 추가**
- 모델이 unnecessarily complex할 때 overfitting이 발생함 → 정말 필요할 때만 항을 갖도록 **regularize**
- 즉, **정말 필요하지 않으면 weight를 작게 (또는 0으로) 강제**

---

### Lasso

#### Ridge Regression의 한계

Ridge regression은 한 가지 명백한 단점이 있음: 

- subset selection과 달리, **모든 $p$개의 predictor를 final model에 포함**시킴. 
- 저차항은 사용하고 고차항은 0으로 만들고 싶은데, 다같이 줄어들고 있음
- Coefficient를 0에 가깝게 만들 뿐, 정확히 0으로 만들진 않음.


#### 해결책 : Lasso


Lasso coefficient는 다음을 최소화함:

$$ \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j| $$

$$ = \underbrace{\sum_{i=1}^{n} (y_i - \beta_0 - x_i^\top \beta)^2}_{\text{RSS}} + \lambda \|\beta\|_1 $$

- $\lambda \geq 0$은 regularization 강도를 조절하는 하이퍼파라미터
- $\beta = [\beta_1, \ldots, \beta_p]^\top \in \mathbb{R}^p$
- $\|\|\beta\|\|_1 = \|\beta_1\| + \cdots + \|\beta_p\|$

---

#### Ridge와의 차이

Ridge regression처럼 coefficient를 0 쪽으로 shrink하지만, **L1 penalty의 특성상 $\lambda$가 충분히 크면 일부 coefficient를 정확히 0으로 만듦**.

- 따라서 lasso는 best subset selection처럼 **variable selection**을 수행하는 경향이 있음. 
- 즉 **sparse model** (적은 수의 variable만 사용하는 모델)을 만들어냄.

cf) $L_p$ loss는:

$$ L_p = \sqrt[p]{\sum_{j=1}^{d} |x_j|^p} $$

$L_1$과 $L_2$는 이 일반화된 loss의 special case임.

<img width="100%" alt="ridge lasso comparison" src="/assets/images/2026-05-25/ridge-lasso.png">

----

#### Lasso가 왜 정확히 0을 만드는가

<img width="50%" alt="lasso area" src="/assets/images/2026-05-25/lasso.png">

<img width="50%" alt="lasso area" src="/assets/images/2026-05-25/lasso-area.png">

기하학적으로 보면, $L_1$ 제약 영역은 **마름모(diamond) 모양**, $L_2$ 제약 영역은 **원**임.

- 최적화 과정에서 unregularized 최적점이 제약 영역 밖에 있다면, 영역 안에서 가장 가까운 점으로 이동해야 함
- 마름모의 경우, 대부분의 점에서 가장 가까운 영역 내 점은 **꼭짓점(vertex)** 임
- 꼭짓점은 $(0, a), (a, 0), (0, -a), (-a, 0)$ 같은 형태이므로, **한 좌표가 0**

즉 외부 점들이 분홍색 영역(꼭짓점에 가까운 영역)에 있으면 한 좌표가 0인 꼭짓점으로 끌려감.

---

#### 고차원에서의 Sparsity

<img width="50%" alt="lasso area in 3d" src="/assets/images/2026-05-25/lasso-3d.png">


차원이 높아지면 lasso는 더 sparse한 모델을 장려함. 3차원의 경우 마름모가 정팔면체(octahedron) 같은 형태가 됨:

- **Vertex**: 3개 좌표 중 2개가 0
- **Edge**: 3개 좌표 중 1개가 0
- **Face**: 모든 좌표가 non-zero

차원이 높을수록 vertex/edge의 비중이 커져서 더 많은 좌표가 0이 됨.

---

#### Ridge Regression, Lasso 비교

- $\lambda$가 극단적으로 클 때 차이가 명확함
  - **Lasso**: bias(외에는) 제외하고 모든 coefficient가 0이 됨
  - **Ridge regression**: 모든 coefficient를 연속적으로 shrink하지만 0으로 만들지는 않음

- **두 방법 중 어느 하나가 universally 더 낫진 않음**
- 일반적으로 **response가 소수의 predictor에만 의존할 때 lasso가 유리**
- 하지만 실제 데이터에서는 response와 관련된 predictor 수를 a priori 알 수 없음
  - lasso는 모델을 sparse 하게 만듦
  - 모든 항을 다 갖고 있는 ridge와 비교했을 때, 어느 게 더 성능이 좋을지는 해봐야 앎
- **Cross-validation**으로 데이터셋마다 어느 방법이 더 나은지 판단해야 함

---

### Selecting the Hyperparameters

Subset selection과 마찬가지로, ridge regression과 lasso 모두 **$\lambda$를 결정하는 방법**이 필요함.

**Cross-validation**이 표준적인 해결책:

1. $\lambda$ 값의 grid를 선택
2. 각 $\lambda$에 대해 cross-validation error rate를 계산
3. CV error가 가장 작은 $\lambda$를 선택
4. (Optional) 선택된 $\lambda$로 **전체 데이터를 사용해 모델을 re-fit**

<img width="80%" alt="selecting hyperparameter lambda" src="/assets/images/2026-05-25/selecting-hyperparameter.png">

Credit 데이터에 ridge regression을 적용한 예시를 보면, CV error가 최소가 되는 지점에서 $\lambda$가 선택되고, 그에 해당하는 coefficient들이 final model의 파라미터가 됨.

---

## Summary of Model Selection
<img width="60%" alt="lasso area" src="/assets/images/2026-05-25/model-dimensions.png">

| 항목                  | Simple 모델    | Complex 모델  |
| --------------------- | -------------- | ------------- |
| **# of parameters**   | Small          | Large         |
| **Model rigidity**    | Rigid          | Flexible      |
| **Model complexity**  | Simple         | Complex       |
| **Decision boundary** | Smooth         | Irregular     |
| **Regularization**    | High $\lambda$ | Low $\lambda$ |
| **Extreme problem**   | Underfitting   | Overfitting   |
| **Bias**              | High           | Low           |
| **Variance**          | Low            | High          |

- **bias와 variance의 균형**을 잘 맞춰서 새로운 데이터에 일반화되는 모델을 만드는 게 목표. 
- 그 도구로 early stopping, ridge regression, lasso, 그리고 이들의 hyperparameter를 결정하는 cross-validation을 사용함.