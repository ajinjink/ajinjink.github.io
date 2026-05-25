---
title: "[ML] Decision Trees"
date: 2026-05-25
categories: ["AI", "ML"]
tags: ["decision tree", "regression", "regression tree", "classification", "MLE", "top-down", "greedy", "tree pruning", "overfitting", "weakest link pruning", "classification tree", "recursive binary splitting", "inpurity", "entropy", "gini index", "regularization", "penalty", "RSS"]
use_math: true
---


### Tree-based Learning

Tree-based 학습은 다음과 같은 특징을 가짐:

- **predictor space를 단순한 영역들로 분할(segmenting)** 함
- 그 분할 규칙들을 **트리 형태로 요약**할 수 있음
- **regression과 classification** 모두에 적용 가능함

<img width="80%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/decision-tree.png">


- **장점**: 단순하고 해석하기 좋음 (interpretable)
- **단점**: 일반적으로 예측 정확도(prediction accuracy)는 경쟁력 없음

- **bagging, random forest, boosting** 같은 ensemble 방법들은 여러 트리를 키워서 하나의 합의된 예측으로 결합하는데, 이게 예측 정확도를 극적으로 끌어올림. 
  - 그래서 트리는 그 자체보다 ensemble의 building block으로서 중요함.

- 해석성-유연성 평면에서 보면 트리는 Generalized Additive Models 근처에 위치함. 
- Subset Selection/Lasso 같은 게 해석성 최상위, SVM/Deep Learning이 유연성 최상위에 있음.

---

## Regression Tree

### 예시: Baseball Players' Salary Data

<img width="50%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/baseball.png">

- Predictor 변수 (2개)
  - `Years`: 메이저리그 경력 연수
  - `Hits`: 전년도 안타 수
- **Target**: 연봉 `Salary` (log scale, $100,000 → 5.0)
- **Task**: regression (log Salary 예측)

### Regression Tree의 동작 방식

<img width="60%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/baseball-tree.png">

입력 $\mathbf{x} \in \mathbb{R}^d$가 주어지면:

1. Root의 조건 (Years < 4.5)을 확인함
   - 참이면 왼쪽 subtree로
   - 거짓이면 오른쪽 subtree로
2. Leaf 노드에 도달할 때까지 반복함
3. Leaf의 값을 $y$ (log Salary)로 출력함

<br>
각 노드의 조건은 $x_j < t_k$ 형태임:

- $x_j$는 입력 $\mathbf{x}$의 예측 변수 중 하나
- $t_k$는 $k$번째 internal 노드의 threshold

<br>
예시 트리(internal 노드 2개, leaf 3개)에서:

- $R_1 = {\mathbf{x} \mid \text{Years} < 4.5}$ → 5.11
- $R_2 = {\mathbf{x} \mid \text{Years} \geq 4.5, \text{Hits} < 117.5}$ → 6.00
- $R_3 = {\mathbf{x} \mid \text{Years} \geq 4.5, \text{Hits} \geq 117.5}$ → 6.74

<img width="60%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/baseball-after.png">

각 leaf의 값은 **해당 영역에 속하는 training 샘플들의 response의 평균**임.

- 경험(Years)이 가장 중요함. 경험이 많을수록 Salary가 높아짐
- 충분한 경험이 있을 때(5년 이상)는 안타 수(Hits)도 영향을 줌. Hits가 많을수록 Salary가 높음

---

## Regression Tree 만들기

### 큰 그림: 2단계 알고리즘

- **Step 1**
    - predictor space ($x_1, x_2, \ldots, x_p$의 가능한 값들)를 $J$개의 distinct하고 non-overlapping한 영역 $R_1, R_2, \ldots, R_J$로 분할함
- **Step 2**
    - 영역 $R_j$에 속하는 모든 관측치에 대해 동일한 예측을 함

---

### Step 2: 각 영역 $R_j$에 어떤 값을 할당할 것인가

가장 단순한 방법은 각 영역에 **상수(constant)** 를 할당하는 것임. 

- 해당 영역에 속하는 모든 예시는 동일한 값 $\hat{\mathbf{y}}_j$로 예측됨. 

- 단순해 보이지만 영역 크기가 충분히 작으면 상수도 합리적인 추정이 됨.

목표는 least squares를 최소화하는 것:

$$ \min \sum_{j=1}^{J} \sum_{i \in R_j} (\mathbf{y}_i - \hat{\mathbf{y}}_j)^2 $$

각 영역 $R_j$에 대해 예측값을 상수값으로 대체:

$$ \hat{\mathbf{y}}_j = \underset{\mathbf{c}}{\operatorname{argmin}} \sum_{i \in R_j} (\mathbf{y}_i - \mathbf{c})^2 $$
<br>

#### MLE로 $\hat{\mathbf{y}}_j$ 유도

region이 나뉘어진 상태에서, 각 region에 할당할 상수값 c는 어떻게 구할 것인가?

목적함수를 $f(\mathbf{c})$로 두면:

$$ \begin{aligned} f(\mathbf{c}) &= \sum_{i \in R_j} (\mathbf{y}_i - \mathbf{c})^2 \\ &= \sum_{i \in R_j} (\mathbf{y}_i^2 - 2\mathbf{c}\mathbf{y}_i + \mathbf{c}^2) \\ &= \sum_{i \in R_j} \mathbf{y}_i^2 - 2\mathbf{c} \sum_{i \in R_j} \mathbf{y}_i + n_j \mathbf{c}^2 \end{aligned} $$

-  $n_j$ : 영역 $R_j$의 training 샘플 수

$f'(\mathbf{c}) = 0$을 풀면:

$$ f'(\mathbf{c}) = -2 \sum_{i \in R_j} \mathbf{y}_i + 2 n_j \mathbf{c} = 0 $$

$$ \therefore \mathbf{c} = \frac{1}{n_j} \sum_{i \in R_j} \mathbf{y}_i $$

- 즉, **각 영역 $j$의 training 샘플 평균**을 $\hat{\mathbf{y}}_j$로 쓰면 됨.

---

### Step 1: 어떻게 $J$개 영역으로 split할 것인가

이상적으로는 RSS를 최소화하는 박스 $R_1, \ldots, R_J$를 찾고 싶음:

$$ \min \sum_{j=1}^{J} \sum_{i \in R_j} (\mathbf{y}_i - \hat{\mathbf{y}}_j)^2 \quad \text{where} \quad \hat{\mathbf{y}}*j = \frac{1}{n_j} \sum*{i \in R_j} \mathbf{y}_i $$

- 그런데 feature space를 $J$개 박스로 가능한 모든 partition을 고려하는 건 **계산상 불가능(computationally infeasible)**
- 모든 조합을 다 계산해 볼 수는 없어

- 그래서 대신 **top-down, greedy approach**, 즉 **recursive binary splitting**을 씀:

  - **Top-down**
    - 트리의 root에서 시작해서 successively predictor space를 split함. 
    - 각 split은 트리에서 두 개의 새 branch로 나타남

  - **Greedy**: 각 단계에서 그 시점의 best split을 함. 
    - 나중 step에서 더 좋은 트리로 이어질 split을 looking ahead하지 않음

1. 처음에는 하나도 안 쪼개진 상태로 시작
2. 각 변수를 기준으로 쪼개고 오차 예측치를 계산
3. 총 오차가 가장 적게 나오는 변수/기준을 하나 선정해서 split.
4. 쪼개진 각 region에 대해서 가능한 경우 다 split 해보고, 나올 수 있는 최적의 기준 선정해서 split
5. 반복

<br>

#### Recursive Binary Splitting의 수식

모든 predictor $x_1, \ldots, x_p$와 cutpoint $s$의 모든 가능한 값을 고려해서, resulting tree의 RSS가 가장 낮아지는 $(j, s)$를 고름.

각 $j, s$에 대해 half-plane 쌍을 정의함:

$$ R_1(j, s) = {\mathbf{x} \mid x_j < s}, \quad R_2(j, s) = {\mathbf{x} \mid x_j \geq s} $$

그리고 다음 RSS를 최소화하는 $j, s$를 찾음:

$$ \sum_{i: x_i \in R_1(j,s)} (y_i - \hat{y}_{R_1})^2 + \sum_{i: x_i \in R_2(j,s)} (y_i - \hat{y}_{R_2})^2 $$

선택된 $j, s$가 조건 $x_j < s$를 결정함.

---

### 과정 반복

- 전체 predictor space를 split하는 대신, **이전에 식별된 영역 중 하나를 split**함
- $t$번째 step에서는 $t$개 영역에서 시작해 $t+1$개 영역으로 끝남
- **stopping criterion**이 만족되면 종료함 (예: 어떤 영역도 5개 이상의 관측치를 갖지 않을 때)

___

### Illustration: 단계별 예시

#### 첫 번째 split

<img width="50%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/baseball-step1.png">

$j \in {\text{Years}, \text{Hits}}$에 대해 각 $(j, s)$ 쌍에 대한 RSS를 계산함. 예시에서는 다음과 같은 값들이 나옴:

- Hits 방향 후보: 12.7, 11.4, 10.5, 13.5, ...
- Years 방향 후보: 5.8, 5.6, ..., 14.2

최솟값 **5.6**을 주는 split이 선택됨.

---

#### 두 번째 split

<img width="50%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/baseball-step2.png">

이제 split된 두 영역 각각 내에서 다시 best $(j, s)$를 찾음. 다시 모든 후보를 평가하면 최솟값 **5.2**가 나오고, 거기서 split함.

---

#### 세 번째 이후

3개의 splittable한 영역들 안에서 best $(j, s)$를 계속 찾아 나감.

**Splittable region**이란 해당 영역이 minimum threshold 이상의 training 샘플을 갖고 있다는 뜻임. 영역이 이 최소값 이하로 떨어지면 그 영역은 fixed되어 더 이상 split 대상이 되지 않음.

<img width="50%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/baseball-step3.png">

---

### Regression Tree 알고리즘 요약

```
while (!stopping condition):
    for all splittable node v, dimension j, threshold s:
        split R1(v, j, s) = v(x_j < s)
        split R2(v, j, s) = v(x_j ≥ s)

        compute y1 = sum(y_i) / |{x: x ∈ R1}| for x_i in R1
        compute y2 = sum(y_i) / |{x: x ∈ R2}| for x_i in R2

        err(v, j, s) = sum((y - y1)^2) for x_i in R1
                     + sum((y - y2)^2) for x_i in R2

    pick (v, j, s) with minimum err(v, j, s)
    split node v at x_j = s
```

- stopping condition : 모든 노드들이 더 이상 쪼갤 수 없을 만큼 적은 수의 데이터가 남아 있을 때

---

## Tree Pruning

### Decision Tree에서의 overfitting

<img width="80%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/overfitting.png">

- Decision Tree에서도 overfitting 일어남 

- 예를 들어 어떤 영역 $R_5$가 단 한 개의 training 점만으로 한 클래스로 예측된다면, 그 큰 영역 전체가 그 한 점 때문에 그 클래스로 예측됨. 

- 이는 심각한 overfitting을 유발함.

<br>

overfitting을 줄이려면 모델을 단순한 쪽으로 가져가야 함. 그런데:

- "파라미터 수 줄이기" → decision tree는 **learnable parameter가 없음**
- "regularization 강화" → decision tree는 **학습 알고리즘이 deterministic**함
- "모델 복잡도 낮추기" → decision tree는 이미 각 영역에서 **상수**라 가장 단순함

남은 방법은 하나: **split 수를 줄여서 결정 경계를 더 smooth하게 만들기**.

---

### Smaller Tree 전략의 문제점

- 작은 트리 (적은 split, 작은 $J$)는 분산을 낮추고 해석을 좋게 함. 
  - 약간의 bias가 늘어나는 비용을 치름.

- 한 가지 방법은 split으로 인한 RSS 감소량이 어떤 (높은) threshold를 넘을 때만 계속 split하는 것임.

<br>
**하지만** 이런 단순한 smaller tree는 **너무 근시안적(short-sighted)** 임:

- 초기에는 가치 없어 보이는 split이 그 뒤에 매우 좋은 split으로 이어질 수 있음
- 트리 구축 과정의 **greedy nature** 때문에 생기는 문제임
- greedy algorithm : 지금 현 상황에서 이득이 가장 큰 선택을 하고, 그 선택을 계속 안고 감

<img width="80%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/greedy-problem.png">

---

### 해결책 : Weakest Link Pruning (Cost Complexity Pruning)

대안은 **큰 트리 $T_0$를 먼저 키우고, 그걸 prune해서 작게 만드는 것**임.

Non-negative tuning parameter $\alpha$에 의해 인덱싱된 트리들의 sequence를 고려함. 

각 $\alpha$값에 대해 다음을 최소화하는 $T \subset T_0$가 존재함:

$$ \sum_{m=1}^{|T|} \sum_{x_i \in R_m} (y_i - \hat{y}_{R_m})^2 + \alpha |T| $$

- $\|T\|$는 트리 $T$의 leaf 노드 수
- $\alpha$는 pruned 트리의 **복잡도와 training data fit 사이의 trade-off**를 조절함
- $\alpha \|T\|$ 항이 regularization의 penalty 항 역할을 함
  - 앞 항을 minimize 하고 싶어 (모든 region의 예측값들에 대한 squared loss)
  - 뒤에 regularization 항 $\alpha \|T\|$을 penalty로 추가 (노드의 개수 = 쪼갠 횟수)
  - 더 작은 트리를 선호하게 됨

- main term에서 **이점(advantage)이 가장 작은 두 leaf 노드를 merge**하는 식으로 쉽게 달성할 수 있음.

- 그래서 이름이 "weakest link" pruning

---

### Tree Building Process with Pruning

전체 절차는 다음 4단계임:

1. **Recursive binary splitting**을 써서 training data에 큰 트리 $T_0$를 키움. 
   - 각 terminal 노드가 어떤 최소 observation 수 미만이 될 때만 멈춤
   - binary splitting 계속 해서, 쪼갤 수 있는 데까지 쪼개
2. **Weakest link pruning**을 $T_0$에 적용해서 $\alpha$의 함수로서 best subtree들의 sequence를 얻음
   - leaf에 있는 두 값을 합했을 때, loss가 가장 덜 나빠지는 걸 찾아서 merge
3. $k$-fold **cross-validation**으로 $\alpha$를 선택함
4. Step 2에서 선택된 $\alpha$ 값에 해당하는 subtree를 반환함

### Pruning 예시 (Baseball Salary Data)

- **Step 1**: 큰 unpruned 트리 $T_0$를 학습함.
  - <img width="80%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/tree-building-step1.png">
- **Step 2**: weakest link pruning을 적용해 leaf들을 점진적으로 merge함
- **Step 3**: cross-validation으로 트리 크기에 따른 MSE를 그려보면, training error는 트리가 커질수록 계속 감소하지만 cross-validation/test error는 어느 시점 이후 다시 올라감.
  - 이 예시에서는 **leaf 3개**에서 best validation 성능을 보임
  - <img width="80%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/tree-building-step3.png">
- **Step 4**: 선택된 크기(3)의 트리를 반환함. 결과는 `Years < 4.5`와 `Hits < 117.5` 두 split만 남은 매우 단순한 트리
  - <img width="60%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/tree-building-step4.png">

---

## Classification Tree

### Regression Tree와의 차이

기본 구조는 거의 같음. 단지:

- regression은 **quantitative response**를 예측함
- classification은 **discrete (categorical) response**를 예측함

Classification에서는 각 관측치가 **속한 영역에서 가장 흔한 클래스(most commonly occurring class)** 에 속한다고 예측함.

알고리즘 두 단계 (Step 1: 영역 분할, Step 2: 영역별 예측)는 regression과 동일함. 차이는 Step 1과 Step 2의 디테일에 있음.

---

### Step 2: 각 영역에 어떤 값을 할당할 것인가

- **Regression**: 상수 $\hat{\mathbf{y}}_j$를 쓰고 least squares를 최소화함

$$ \hat{\mathbf{y}}_j = \underset{\mathbf{c}}{\operatorname{argmin}} \sum_{i \in R_j} (\mathbf{y}_i - \mathbf{c})^2 \implies \mathbf{c} = \frac{1}{n_j} \sum_{i \in R_j} \mathbf{y}_i $$

- **Classification**: 각 영역에서 **majority vote**를 씀

$$ \hat{\mathbf{y}}_j = \underset{k}{\operatorname{argmax}}  p_{j,k} = \underset{k}{\operatorname{argmax}} \frac{1}{n_j} \sum_{\mathbf{x}_i \in R_j} I(\mathbf{y}_i = k) $$

- $n_j$는 영역 $j$의 training 샘플 수
- $p_{j,k}$는 영역 $j$ 내에서 클래스 $k$가 차지하는 비율
- Majority vote
  - 해당 region에 속하는 training data들에 대해서, { label이 k인 개수 / 전체 개수 }를 구해서 region에서 비율이 가장 높은 k를 선택
  - classification error가 가장 작아지는 값을 하나의 대표값으로 정하는 것

---

### Step 1: 어떻게 $J$개 영역으로 split할 것인가

- 여전히 **top-down, greedy, recursive binary splitting**을 씀. 

- 그 시점에서 benefit이 가장 커 보이는 노드를 고름.

- 모든 predictor $x_1, \ldots, x_p$와 모든 가능한 cutpoint $s$를 고려해서, **classification error rate가 가장 낮은 결과 트리**가 나오는 predictor와 cutpoint를 선택함.

Half-plane 쌍은 동일함:

$$ R_1(j, s) = {\mathbf{x} \mid x_j < s}, \quad R_2(j, s) = {\mathbf{x} \mid x_j \geq s} $$

그리고 다음 양을 **최소화**하는 $j, s$를 찾음:

$$ \frac{1}{n_1} \text{Impurity}(R_1(v, j, s)) + \frac{1}{n_2} \text{Impurity}(R_2(v, j, s)) $$

- 결과 영역 $R_1, R_2$를 **가능한 한 unanimous(만장일치)** 하게 만드는 것이 목표
- $n_1, n_2$로 나누어주는 이유는 **size로 나누어줘서 큰 영역을 split하는 걸 선호**하기 때문


<br>
**Impurity**

- "한 region 안에 클래스들이 얼마나 섞여있는지"를 재는 양
- region을 하나 던져주면, 클래스가 얼마나 섞여있는지 숫자로 뱉어줌 (얼마나 pure하지 못한지)

- impurity = 0 : 데이터가 극명하게 나뉘어져 있다면 쪼갰을 때 각각의 region이 unanimous하게 하나의 클래스에 속함
- 좋은 split = 자식 region들이 각각 pure해지는 split



---

### Impurity measure 1: Entropy

영역 $R_j$의 impurity는 어떻게 정의할까. 

첫 번째 방법은 **entropy**:

$$ \text{Impurity}(R_j) = -\sum_{k=1}^{K} p_{j,k} \log p_{j,k} $$

- $p_{j,k}$는 영역 $j$ 내에서 클래스 $k$가 차지하는 비율

- $[1, 0, 0, 0, 0]$ → $-1 \log 1 = 0$
- $[0.5, 0.5, 0, 0, 0]$ → $-2 \cdot 0.5 \log 0.5 = \log 2$
- $[0.2, 0.2, 0.2, 0.2, 0.2]$ → $-5 \cdot 0.2 \log 0.2 = \log 5$

<img width="50%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/entropy.png">

- 모든 예시가 단 하나의 클래스에서 나옴 (purest) → entropy = 0 (최솟값)
- 모든 예시가 uniformly 분포함 → entropy = $\log K$ (최댓값)

---

### Impurity measure 2: Gini Index

두 번째로 널리 쓰이는 impurity metric은 **Gini index**임:

$$ G = \sum_{k=1}^{K} p_{j,k} (1 - p_{j,k}) $$

- $[1, 0, 0, 0, 0]$ → $0$
- $[0.5, 0.5, 0, 0, 0]$ → $2 \cdot \frac{1}{4} = 0.5$
- $[0.2, 0.2, 0.2, 0.2, 0.2]$ → $5 \cdot 0.2 \cdot 0.8 = 0.8$

<img width="50%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/gini.png">

- Entropy와 Gini index는 **수치적으로 매우 비슷함**. 

- 둘 다 $p = 0.5$ 근처에서 최댓값을 가지고 $p = 0$ 또는 $p = 1$에서 0이 되는 종 모양 곡선임. 

- Entropy가 조금 더 위쪽에 있고 Gini index가 조금 아래쪽에 있을 뿐임.

---

### Classification Tree 예시: Heart Data

- 303명 chest pain 환자의 binary outcome `HD`
- `Yes`: angiographic test 기반 심장병 있음 / `No`: 없음
- Predictor 13개 (Age, Chol, 기타 심폐 기능 측정치 등)

<img width="80%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/heart-data.png">

학습된 unpruned 트리는 Thal, Ca, MaxHR, ChestPain, RestBP, Chol, Slope, Age, Oldpeak, RestECG, Walks, Sex 등 다양한 변수를 쓰는 깊은 트리가 됨.

<img width="80%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/heart-pruned.png">

Pruning 결과 cross-validation error가 가장 낮은 지점이 선택되어 훨씬 단순한 트리가 됨.

---

### Decision Boundary 비교

<img width="50%" alt="validation curve" src="/assets/images/2026-05-25/decision-tree/decision-boundary.png">

- linear model은 깔끔한 직선 경계를, decision tree는 계단식의 axis-aligned 경계를 만듦. 
- Data 1 (선형 경계가 자연스러운 데이터)
  - Tree는 부자연스러움
- Data 2 (axis-aligned 영역이 자연스러운 데이터)
  - linear model은 잘 못 맞추지만, decision tree는 깔끔하게 영역을 잡아냄

- tree는 **axis-aligned rectangular partition**이 자연스러운 데이터에 강하고, **oblique한 경계**가 필요한 데이터에는 약함.

---

### Classification Tree 알고리즘 요약

```
while (!stopping condition):
    for all splittable node v, dimension j, threshold s:
        split R1(v, j, s) = v(x_j < s)
        split R2(v, j, s) = v(x_j ≥ s)

        compute y1 = argmax_k(y_i = k) for x_i in R1
        compute y2 = argmax_k(y_i = k) for x_i in R2

        err(v, j, s) = impurity(R1)/|{x: x ∈ R1}|
                     + impurity(R2)/|{x: x ∈ R2}|

    pick (v, j, s) with minimum err(v, j, s)
    split node v at x_j = s
```

Regression tree의 알고리즘과 거의 동일함. 차이는:

- prediction: 평균 → argmax (majority vote)
- error: squared error → impurity (entropy or Gini)

---

## Pros and Cons of Decision Trees

### Pros (장점)

- 사람에게 설명하기 매우 쉬움 (easy to explain)
  - 어떤 사람들은 decision tree가 대부분의 다른 regression/classification 방법보다 **인간의 의사결정에 더 가깝다**고 믿음
  - 트리는 graphically 표시할 수 있고, 작을 때는 **비전문가도 쉽게 해석**할 수 있음
- **Qualitative predictor를 쉽게 다룰 수 있음**.
  - Dummy variable을 만들 필요가 없음

### Cons (단점)

- **예측 정확도(predictive accuracy)** 가 다른 regression/classification 방법들만큼 좋지 않음
- **Robust하지 않음**. 데이터의 작은 변화가 최종 학습된 트리에 큰 변화를 일으킬 수 있음

- 이 단점들은 ensemble 방법들(bagging, random forests, boosting)로 상당 부분 해결됨. 
- 트리는 그 자체보다 building block으로서 더 빛난다는 것이 결론.