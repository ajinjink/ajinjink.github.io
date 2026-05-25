---
title: "[ML] Bayesian Optimization"
date: 2026-05-20
categories: ["AI", "ML"]
tags: ["Bayes' theorem", "Gaussian process", "GP regression", "Bayesian optimization", "acquisition function", "GP fit", "Cholesky decomposition", "conjugate"]
use_math: true
---

## 요약

- 어떤 목적함수가 있어 (e.g. 머신러닝 모델의 loss function).
    - 그런데 이 목적함수가 블랙박스이거나 실행하는 데 코스트가 너무 높아.
    - 그래서 목적함수를 모방하는 surrogate model을 하나 둬 = GP
- Gaussian Process
    - 함수 공간 위의 분포
    - 임의의 유한한 점들 ${x_1, \dots, x_n\}$에서의 함숫값 벡터가 multivariate Gaussian을 따르도록 정의되는 stochastic process
    - prior로 mean function $m(x)$와 kernel $k(x, x')$을 정의하면 함수에 대한 prior가 정해져
- GP fit
    - kernel의 hyperparameter 학습
        - kernel의 모양을 데이터에 맞추기
    - posterior 계산
        - GP prior를 데이터 $D$에 conditioning 해서 posterior GP 얻어
        - posterior GP는 함수 공간 위의 분포로 존재
            - 필요한 $x^\*$가 들어오면, 그 점에서의 $\mu(x^\*), \sigma^2(x^\*)$ 뽑을 수 있어
- Acquisition function으로 $x_\text{next}$ 추출
    - 후보 $x^\*$들에 대해 $\mu(x^\*), \sigma^2(x^\*)$ 평가하면서 acquisition function 최대화
    - exploitation + exploration
- $x_\text{next}$를 실제 목적함수로 evaluate
    - $y_\text{next}​=f(x_\text{next​})$
- {$x_\text{next}, y_\text{next}$}를 데이터 $D$에 추가
- GP fit ~ {$x_\text{next}, y_\text{next}$} 과정 반복
    - 점점 실제 목적함수를 더 잘 모방하는 surrogate model이 됨
    - 실제 목적함수를 덜 실행하면서 최적화. sample efficiency


----


## 베이지안 통계

### 빈도주의 vs 베이지안

확률을 해석하는 두 가지 큰 흐름:

- **빈도주의(Frequentist)**: 확률을 "사건을 무한히 반복했을 때의 상대빈도"로 봄
    - 모수(parameter) $\theta$는 고정된 미지의 값이고, 데이터가 랜덤
- **베이지안(Bayesian)**: 확률을 "어떤 명제에 대한 믿음의 정도(degree of belief)"로 봄
    - 데이터를 보기 전 우리는 $\theta$에 대해 어떤 믿음을 가지고 있고, 데이터를 보고 나서 그 믿음을 업데이트
    - 즉 **$\theta$ 자체가 확률변수**처럼 다뤄짐

베이지안의 가장 큰 장점은 "내가 모르는 것에 대한 불확실성"을 분포로 자연스럽게 표현할 수 있다는 점.

---

### 베이즈 정리(Bayes' Theorem)

조건부 확률의 정의에서 출발한다.

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B \mid A) = \frac{P(A \cap B)}{P(A)}
$$

두 식에서 $P(A \cap B)$를 소거하면

$$
P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta)\, P(\theta)}{P(\mathcal{D})}
$$


- $P(\theta)$: **사전분포(prior)** — 데이터를 보기 전 $\theta$에 대한 믿음
- $P(\mathcal{D} \mid \theta)$: **우도(likelihood)** — 어떤 $\theta$가 주어졌을 때 우리가 본 데이터가 나올 가능성
- $P(\theta \mid \mathcal{D})$: **사후분포(posterior)** — 데이터를 보고 난 뒤 업데이트된 믿음
- $P(\mathcal{D}) = \int P(\mathcal{D}\mid\theta) P(\theta)\, d\theta$: **주변우도(marginal likelihood, evidence)** — $\theta$를 모두 적분해서 없앤 데이터의 확률. $\theta$에 의존하지 않는 정규화 상수.

실제 계산에서는 상수를 떼고

$$
P(\theta \mid \mathcal{D}) \propto P(\mathcal{D} \mid \theta)\, P(\theta)
$$

만 다루는 경우가 많음


#### 직관적 해석

> **사후분포 $\propto$ 우도 $\times$ 사전분포**

"처음 믿음(prior)"에 "데이터가 말해주는 것(likelihood)"을 곱해서 "업데이트된 믿음(posterior)"을 얻음.
데이터가 많아질수록 우도가 사전분포를 압도하고, 데이터가 적을수록 사전분포의 영향이 큼.

---

#### 간단한 예제: 동전 던지기

앞면 확률을 $\theta \in [0,1]$라 하자. 동전을 $n$번 던져 앞면이 $k$번 나왔다.

- 우도: $P(\mathcal{D}\mid\theta) = \theta^k (1-\theta)^{n-k}$ (이항 분포)
- 사전: $\theta \sim \text{Beta}(\alpha, \beta)$, 즉 $P(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}$

곱하면

$$
P(\theta \mid \mathcal{D}) \propto \theta^{k+\alpha-1}(1-\theta)^{n-k+\beta-1}
$$


$\theta \mid \mathcal{D} \sim \text{Beta}(\alpha+k,\, \beta+n-k)$
- 이것도 베타 분포

이렇게 사전과 사후가 같은 분포족이 되는 경우 : **conjugate prior**
- 베이지안에서 계산이 깔끔하게 떨어지는 대표적인 사례

---

## Gaussian Process Regression (GPR, 가우시안 프로세스 회귀)

베이지안 최적화는 "내가 모르는 함수 $f(x)$"에 대한 사후분포가 필요.
그 사후분포를 만드는 가장 흔한 도구가 **가우시안 프로세스(Gaussian Process, GP)**.

### 다변량 정규분포

$n$개의 변수가 결합 정규분포를 따른다는 것은

$$
\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu},\, \boldsymbol{\Sigma})
$$

평균 벡터 $\boldsymbol{\mu}$와 공분산 행렬 $\boldsymbol{\Sigma}$로 완전히 정해진다는 뜻이다.
다변량 정규분포의 핵심 성질은 **conditioning이 또 정규분포**라는 것.

두 부분 $\mathbf{y}_1, \mathbf{y}_2$로 나누면

$$
\mathbf{y}_1 \mid \mathbf{y}_2 \sim \mathcal{N}\!\left(\boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{y}_2 - \boldsymbol{\mu}_2),\; \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}\right)
$$

이 닫힌 형태의 conditioning에서 GP regression이 일어남.

---

### Gaussian Process

Gaussian Process는 **"함수에 대한 분포"**.

어떤 입력 점들의 유한 부분집합 $\{x_1, \dots, x_n\}$을 잡아도, 그 점들에서의 함수값 $(f(x_1), \dots, f(x_n))$이 결합 정규분포를 따른다.

$$
f(x) \sim \mathcal{GP}\big(m(x),\, k(x, x')\big)
$$


- $m(x) = \mathbb{E}[f(x)]$: 평균 함수 (보통 0으로 둔다)
- $k(x, x') = \mathrm{Cov}(f(x), f(x'))$: **공분산 함수(커널, kernel)**

다변량 정규분포가 평균 벡터와 공분산 행렬로 정해지는 것의 무한 차원 일반화라고 생각하면 됨

---

### kernel

kernel : 함수의 "성격"을 정하는 부분

가장 많이 쓰는 것이 **RBF 커널(SE kernel, squared exponential)** :

$$
k(x, x') = \sigma_f^2 \exp\!\left( -\frac{\|x - x'\|^2}{2\ell^2} \right)
$$

- $\ell$ (length-scale): 함수가 얼마나 천천히 변하는가. $\ell$이 크면 부드럽고, 작으면 거칠게 출렁인다.
- $\sigma_f^2$: 함수값의 전체적인 크기(분산).

직관적으로 **"입력이 가까우면 출력도 비슷하다"**는 가정을 수학적으로 표현한 것.
- 이 외에도 Matérn, periodic, linear 커널 등이 있고, 도메인 지식에 따라 변화하여 사용

---

### GP Regression: 예측 공식

훈련 데이터 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$이 있고, $y_i = f(x_i) + \varepsilon_i,\ \varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$ 라 하자. 

새 점 $x_\*$에서의 예측값 $f_\*$를 알고 싶다.

GP 가정에 따라

$$
\begin{bmatrix} \mathbf{y} \\ f_* \end{bmatrix}
\sim \mathcal{N}\!\left( \mathbf{0},\; \begin{bmatrix} K + \sigma_n^2 I & \mathbf{k}_* \\ \mathbf{k}_*^\top & k_{**} \end{bmatrix} \right)
$$


- $K_{ij} = k(x_i, x_j)$ ($n \times n$ 그램 행렬)
- <span>$$\mathbf{k}_{*} = [k(x_1, x_{*}), \dots, k(x_n, x_{*})]^\top$$</span>
- <span>$$k_{**} = k(x_{*}, x_{*})$$</span>

다변량 정규분포의 conditioning 공식을 그대로 쓰면, **사후분포가 닫힌 형태로 나온다**:

$$

\begin{aligned}
\mu(x_*) &= \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y} \\[4pt]
\sigma^2(x_*) &= k_{**} - \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{k}_*
\end{aligned}

$$

- $\mu(x_*)$: 그 점에서의 **예측 평균**
- $\sigma^2(x_*)$: 그 점에서의 **예측 분산(불확실성)**

### GP의 핵심 직관

- 데이터가 가까이 있는 곳에서는 $\sigma(x_*)$가 작다 → 자신 있게 예측
- 데이터가 멀리 있는 곳에서는 $\sigma(x_*)$가 크다 → 불확실하다
- GP는 점별로 "내가 얼마나 자신 있는지"까지 알려줌

---

## Bayesian Optimization


목표:

$$
x^* = \arg\max_{x \in \mathcal{X}} f(x)
$$

단, $f$는

- **블랙박스**: 미분 정보가 없다. 함수의 식을 모른다.
- **평가 비용이 크다**: 한 번 $f(x)$를 계산하는 데 시간/돈/사람이 많이 든다.
- **노이즈가 있을 수 있다**.
- ex. 하이퍼파라미터 튜닝(한 번 학습에 몇 시간), A/B 테스트, 신소재 합성, 로봇 정책 튜닝, 비싼 시뮬레이션. 그라디언트 디센트나 랜덤 서치로 마구잡이로 돌리기엔 너무 비싸다



진짜 함수 $f$ 대신, 우리가 **저렴하게 다룰 수 있는 대리 모델(surrogate model)**을 만들고, 그 surrogate model을 통해 **"다음에 어디를 평가하면 가장 이득일까?"**를 결정.
- surrogate model로는 GP를 가장 흔히 쓰고, "어디를 평가할까"는 **acquisition function**으로 결정

---

### 알고리즘 전체 흐름

```
1. 초기 점 몇 개를 sample해서 (x_i, y_i) 수집 (LHS/Sobol)
2. while 예산 남음:
     a. 지금까지의 데이터로 GP를 fit            ← 무거운 작업이 여기 다 있음
     b. acquisition function a(x)를 계산
     c. x_{next} = argmax_x a(x)
     d. y_{next} = f(x_{next}) 평가              ← 비싼 블랙박스 호출
     e. 데이터에 (x_{next}, y_{next}) 추가
3. 가장 좋은 관측값을 반환
```

여기서 **(c) "argmax of acquisition"**과 **(a) "GP fit"**이 BO의 두 기둥이다.

---

### Step 1 — 초기 샘플링

GP는 빈 데이터로 fit할 수 없으므로 최초의 $n_{\text{init}}$개 점이 필요.

- **개수**: 보통 $n_{\text{init}} = 5d$ 또는 $10d$ ($d$는 입력 차원)
    - 너무 적으면 GP가 엉터리, 너무 많으면 BO를 쓰는 의미가 없음
- **분포**: 랜덤 균등 추출보다 **LHS / Sobol** 같은 space-filling이 거의 항상 나음. 균등 랜덤은 우연히 한쪽에 몰리는 경우가 자주 생김.
- **목적**: 입력 공간에 대한 "최소한의 시야"를 확보하는 것
    - 이 단계에서 큰 영역을 빈 채로 두면 GP의 분산 $\sigma(x)$가 그 영역에서 영원히 의미 있게 줄지 않아 acquisition이 거기로 끌려갈 수 있다(좋은 의미로든 나쁜 의미로든).

------

### Step 2(a) — GP Fit

GP를 fit한다 :

#### 1. 데이터 전처리

- **출력 정규화**: $y$를 평균 0, 분산 1로 표준화.
    - GP의 prior mean을 0으로 둘 거라면 사실상 필수.
    - 라이브러리들은 `normalize_y=True` 같은 옵션으로 자동 처리.
- **입력 스케일링**: 각 차원을 $[0,1]^d$ 또는 $[-1,1]^d$로 맞춰.
    - 그래야 length-scale prior가 차원별로 합리적인 범위에서 시작하고, 수치적으로도 안정적
- 차원마다 스케일이 천차만별이면(예: 학습률 $10^{-5}\sim 10^{-1}$, 배치 크기 $32\sim 512$) 로그 스케일로 변환한 뒤 표준화

#### 2. 커널과 학습 대상 파라미터 정하기

실무 표준은 RBF 또는 **Matérn-5/2** 커널이다. Matérn-5/2는 RBF보다 덜 부드럽기 때문에(2번 미분 가능 vs 무한 미분 가능) 실제 응용에서 더 robust한 경우가 많다 — BoTorch의 기본값.

각 차원마다 length-scale을 따로 두는 **ARD (Automatic Relevance Determination)** 변형을 쓰면

$$ k(x, x') = \sigma_f^2 \exp!\left(- \sum_{j=1}^{d} \frac{(x_j - x'_j)^2}{2 \ell_j^2} \right) $$

학습할 하이퍼파라미터 $\theta$는

$$ \theta = (\ell_1, \ldots, \ell_d,\ \sigma_f^2,\ \sigma_n^2) $$

총 $d+2$개.

$\ell_j$가 큰 차원은 출력에 별 영향을 안 주는 차원이라는 뜻이고, 작으면 민감한 차원이라는 뜻 — ARD가 자동으로 "중요한 변수"를 찾아주는 효과.

#### 3. 하이퍼파라미터 학습: log marginal likelihood 최대화

$\theta$를 어떻게 정할 것인가?

**데이터의 marginal likelihood**(주변우도)를 최대화 ($f$를 적분으로 없앤 데이터의 확률)

$$ \log p(\mathbf{y} \mid X, \theta) = \underbrace{-\tfrac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y}}_{\text{(i) 데이터 fit}} \underbrace{-\tfrac{1}{2}\log|K_y|}_{\text{(ii) 복잡도 penalty}} \underbrace{-\tfrac{n}{2}\log(2\pi)}_{\text{상수}} $$

- $K_y = K(X, X; \theta) + \sigma_n^2 I$

세 항의 해석이 핵심:

- **(i) 데이터 fit**: 데이터를 잘 설명하지 못하면 이 항이 매우 음수가 된다. 커진다.
- **(ii) 모델 복잡도 penalty (Occam's razor)**: $\log\|K_y\|$는 모델이 표현할 수 있는 함수의 "부피"에 비례
    - 너무 wiggly한(작은 $\ell$) 모델이면 이 항이 자동으로 페널티를 먹임
    - 정칙화를 따로 안 해도 됨
- (iii)는 $\theta$와 무관한 상수.

$\therefore$ marginal likelihood는 "잘 맞추되 너무 복잡해지지 말라"를 자동으로 균형을 잡음.
<br/>

##### 구현

$\log p(\mathbf{y}\mid X,\theta)$는 $\theta$에 대해 미분 가능하므로 L-BFGS, Adam 등으로 최적화. 단,

- **Non-convex**
    - 여러 local optimum이 있음
    - 그래서 보통 **5~10번의 random restart**를 돌려서 best를 고름
- 하이퍼파라미터에 priors를 두고 MAP을 찾는 변형(weakly-informative half-normal on $\ell_j$ 등)도 흔함
    - 데이터가 적을 때 도움이 됨
- 더 들어가면 fully Bayesian GP — $\theta$도 적분해서 없애버리는 방식 (NUTS/HMC)
    - BoTorch의 SAASBO가 이쪽 계열.

#### 4. Cholesky 분해와 예측 계산

$\theta$가 정해졌으면 $K_y^{-1}$을 진짜로 역행렬 계산하지 않고, 더 빠르고 안정적인 **Cholesky 분해**를 사용

$$ K_y = L L^\top $$

- $L$은 하삼각 행렬
- 그러면 $\alpha = K_y^{-1}\mathbf{y}$는
    1. $L, \mathbf{z} = \mathbf{y}$를 forward substitution으로 풀어 $\mathbf{z}$
    2. $L^\top, \alpha = \mathbf{z}$를 backward substitution으로 풀어 $\alpha$

이 $\alpha$가 손에 들어오면, **새 점 $x_\*$에서의 예측은 사실상 공짜**:

$$
\begin{aligned}
\mu(x_{*}) &= \mathbf{k}_{*}^\top \alpha \quad &&O(n) \\[4pt]
\mathbf{v} &= L^{-1} \mathbf{k}_{*} \quad &&\text{(triangular solve)} \\[4pt]
\sigma^2(x_{*}) &= k_{**} - \mathbf{v}^\top \mathbf{v} \quad &&O(n^2)
\end{aligned}
$$

또한 log marginal likelihood의 $\log\|K_y\|$도 $2\sum_i \log L_{ii}$로 싸게 구할 수 있음

**비용 정리**:

| 작업                    | 비용             |
| ----------------------- | ---------------- |
| Cholesky 한 번          | $O(n^3)$         |
| 한 점에서 평균 예측     | $O(n)$           |
| 한 점에서 분산 예측     | $O(n^2)$         |
| $N$개 후보 점 전체 평가 | $O(N \cdot n^2)$ |

- BO에서 $n$은 보통 수십~수백 수준이라 이 비용이 문제가 안 됨
- $n > 1000$ 넘어가면 sparse GP, inducing points, GPU(GPyTorch) 같은 게 등장

#### 5. 수치 안정성

GP에서 가장 흔한 실패 모드는 $K_y$가 수치적으로 positive definite가 아니게 되는 것.

- **Jitter**: 대각선에 $\epsilon I$ ($\epsilon \sim 10^{-6}$)를 더해 PSD를 보장. 라이브러리 기본값.
- **중복/거의 같은 점**: $x_i \approx x_j$이면 $K$의 두 행이 거의 같아져서 조건수가 폭발. → 중복 제거 또는 jitter를 키워라.
- **노이즈 분산 하한**: 무노이즈 함수라 해도 $\sigma_n^2$를 $10^{-6}$ 정도로 두면 안정적.
- **Cholesky 실패 시 재시도**: jitter를 $10^{-6} \to 10^{-4} \to 10^{-2}$로 키우면서 다시 시도하는 retry 로직이 표준.

------

### Step 2(b)(c) - Acquisition 계산과 최대화

GP가 fit됐다면 임의의 $x$에서 $\mu(x), \sigma(x)$를 싸게 얻을 수 있음.
그러면

$$ a(x) = \mu(x) + \sqrt{\beta}\,\sigma(x) $$

같은 acquisition을 계산하고, $x_{\text{next}} = \arg\max_x a(x)$를 풀 수 있음.


---



#### UCB Acquisition Function

**Upper Confidence Bound (UCB)** acquisition function은

$$
a_{\text{UCB}}(x) = \mu(x) + \beta^{1/2}\, \sigma(x)
$$

(최소화 문제라면 $\mu(x) - \beta^{1/2}\sigma(x)$의 LCB가 되고, 같은 논리)

- $\mu(x)$: GP가 그 점에서 예측한 평균 → **exploitation**(좋아 보이는 곳)
- $\sigma(x)$: GP가 그 점에서 가진 불확실성 → **exploration**(잘 모르는 곳)
- $\beta$: 두 항의 균형을 조절하는 가중치

---

#### Exploration vs Exploitation tradeoff

- $\beta$가 **작으면**: $\mu(x)$가 큰 점만 골라 감
    - 지금까지 본 최고점 근처만 파게 됨 → local optimum에 빠질 수 있음
- $\beta$가 **크면**: 불확실성이 큰 곳을 더 자주 탐색 → 글로벌 탐색은 잘 되지만 수렴이 느림


#### $\beta$의 선택

가장 단순한 선택은 상수 (예: $\beta = 2$ 또는 $\beta = 4$).

이론적으로는 GP-UCB 논문(Srinivas et al., 2010)에서 누적 후회(cumulative regret)에 대한 sublinear 경계를 보장하는 스케줄을 제시했음:

입력 공간이 유한 후보 집합 $\|\mathcal{X}\| = D$일 때

$$
\beta_t = 2 \log\!\left( \frac{D\, t^2 \pi^2}{6\delta} \right)
$$

처럼 $t$(iteration)에 따라 천천히 커짐.

실무에서는 보통 $\beta \in [1, 10]$ 범위의 상수로 두는 경우가 많고, 위 식대로 스케줄링하면 점점 exploration이 강해짐.

#### 다른 acquisition function과 비교

- **Expected Improvement (EI)**: $a_{\text{EI}}(x) = \mathbb{E}[\max(0, f(x) - f^+)]$
    - 현재 최고값 $f^+$을 얼마나 더 넘을지 기대값
    - 닫힌 형태로 계산 가능
- **Probability of Improvement (PI)**: $f(x) > f^+$일 확률
    - 종종 너무 exploit하는 경향이 있을 수도
- **Thompson Sampling**: 사후분포에서 함수 하나를 샘플링하고, 그것의 argmax를 선택

UCB의 장점은 **해석이 매우 직관적**이고, **이론적 후회 경계가 깔끔**하다는 것.
단점은 $\beta$ 튜닝이 필요하다는 점.

---


------

#### Acquisition 최대화 — 후보점 + 국소탐색

이제 실무에서 자주 놓치는 부분 : **"acquisition function을 어디서 평가해서 argmax를 찾을 것인가?"**

BO에는 두 종류의 "샘플"이 있어서 헷갈리기 쉬움
- 실제로 $f$를 평가할 점들 (비싼 평가) — Step 1, Step 2(c)에서 알고리즘이 고름
- acquisition을 평가할 후보 점들 (저렴한 평가) — argmax를 찾으려고 $a(x)$를 계산해보는 점들

여기서 다루는 "샘플링"은 후자.

##### 1차원: 단순 그리드면 충분

1D 입력 공간 $[a, b]$에서는

```python
import numpy as np
X_grid = np.linspace(a, b, 1000).reshape(-1, 1)
mu, sigma = gp.predict(X_grid, return_std=True)
ucb = mu + beta**0.5 * sigma
x_next = X_grid[np.argmax(ucb)]
```

1000개 정도면 보통 충분.

##### 다차원: 차원의 저주

차원이 $d$일 때 각 축에 $N$개의 격자점을 두면 총 $N^d$개

| $d$ | 축당 100점 | 총 점 수 |
|-----|------------|----------|
| 1   | 100        | 100      |
| 2   | 100        | 10,000   |
| 5   | 100        | $10^{10}$ |
| 10  | 100        | $10^{20}$ |

- $d \geq 4$만 되어도 단순 격자는 망함
- → quasi-random / space-filling design 필요

##### Quasi-random / Space-filling design

후보 집합을 **공간을 골고루 덮는** 방식으로 만듦. 주로 셋 중 하나:

- **Latin Hypercube Sampling (LHS)** : 각 차원을 $N$개의 구간으로 나누고, 각 구간이 정확히 한 번씩 쓰이도록 점을 뽑음
    - 균등한 marginal 분포 보장
- **Sobol Sequence** (low-discrepancy quasi-random) : 어떤 작은 부분 공간을 봐도 균일하게 채우는 점들
    - 가장 추천. `scipy.stats.qmc.Sobol`
- **Halton Sequence** : Sobol과 비슷한 quasi-random 수열

```python
from scipy.stats.qmc import Sobol
sampler = Sobol(d=dim, scramble=True, seed=0)
X_cand = sampler.random(n=4096)        # [0,1]^d
X_cand = lb + (ub - lb) * X_cand        # 실제 범위로 스케일
```

- 후보 개수는 보통 $1{,}000 \sim 10{,}000$ 정도
- 차원이 높을수록 더 많이

##### 실무 표준: 후보점 + L-BFGS-B

후보 집합에서 acquisition을 평가해 best 몇 개를 고른 뒤, 그 점들을 시작점으로 두고 **L-BFGS-B**로 국소 최적화

```
1. Sobol로 4096개 후보 추출
2. acquisition 값 상위 K=10개 점 선택
3. 각 점에서 L-BFGS-B로 a(x) 국소 최대화
4. 가장 좋은 결과를 x_next로 채택
```

- acquisition $a(x)$는 GP에서 미분 가능 → 그라디언트 기반 국소 최적화 가능
- BoTorch, GPyOpt 등 라이브러리의 내부 동작
- 직접 구현 귀찮으면 **BoTorch / Ax / scikit-optimize / GPyOpt** 사용

##### Acquisition은 미분 가능한가?

- UCB, EI 모두 $\mu(x), \sigma(x)$의 함수이고, $\mu, \sigma$가 $x$에 대해 미분 가능 → acquisition도 미분 가능
- 그래서 그라디언트 기반 최적화 가능
- 단, GP **하이퍼파라미터** $\theta$에 대한 그라디언트는 marginal likelihood 최적화에서 쓰는 거고, acquisition을 최대화할 때 쓰는 그라디언트는 **입력 $x$**에 대한 그라디언트로 **별개의 계산**



------

### Step 2(d) — $f(x_{\text{next}})$ 평가

두 가지 변형이 자주 등장:
- **노이즈 있는 평가**: 같은 $x$에서 호출해도 다른 값이 나옴. GP의 $\sigma_n^2$가 이걸 잡아낸다.
- **병렬 평가(batch BO)**: GPU 8개 / 워커 여러 개로 동시에 $q$개 점을 평가하고 싶다면 한 번에 $q$개를 골라야 한다. **q-EI, q-UCB** 같은 변형이 있고, 가장 단순한 방법은 "fantasizing" — 이미 고른 점에서 GP의 평균 예측을 임시 관측값으로 넣고 다음 점을 또 고르는 식.
- 계산 오래 걸림

------

### Step 2(e) — 데이터 업데이트 + 다음 iteration

- 데이터 $\mathcal{D}\_{t+1} = \mathcal{D}\_t \cup \lbrace(x\_{\text{next}}, y\_{\text{next}})\rbrace$
- **GP 재학습**: 보통은 매 iteration마다 처음부터 다시 fit
    - $n$이 작아서 $O(n^3)$이 부담스럽지 않음
- **Incremental update**: $n$이 수백 넘어가기 시작하면 rank-1 Cholesky update로 새 점만 추가해 incremental하게 갱신 가능 ($O(n^2)$)
    - BoTorch 등은 내부적으로 처리.
- **하이퍼파라미터는 매번 다시 학습**
    - 한 번 안정적으로 수렴한 뒤로는 매 iteration마다 새로 학습할 필요는 없고, 몇 iteration에 한 번씩 재학습하는 휴리스틱도 흔함

------

### 종료 조건

- 평가 예산 소진 (가장 흔함)
- 일정 횟수 동안 best 값 개선 없음 (patience)
- 시간/돈 제약 도달
- acquisition 최대값이 어떤 임계 아래로 떨어짐 (이론적으로는 깔끔하지만 실무에선 잘 안 씀)

------

### 요약

- **GP fit (Step 2a)** = 전처리 → 커널 선택 → marginal likelihood로 $\theta$ 학습 (L-BFGS + restarts) → Cholesky로 $\mu, \sigma$ 예측 인프라 구축
    - 한 iteration의 **수치적 비용**이 다 모이는 곳
- **Argmax of acquisition (Step 2c)** = $\mu, \sigma$를 어떻게 섞을지(UCB의 $\beta$) + 어떻게 최대화할지(Sobol + L-BFGS-B)
    - 한 iteration의 **탐색 전략**이 결정되는 곳






---

## 큰 그림 정리

```
              ┌─────────────────────┐
              │  비싼 블랙박스 f(x) │
              └──────────┬──────────┘
                         │ 평가 (적게)
                         ▼
              ┌─────────────────────┐
              │ 데이터 (x_i, y_i)   │
              └──────────┬──────────┘
                         │ fit
                         ▼
       ┌──────────────────────────────────┐
       │ GP: μ(x), σ(x)                   │
       │  (사후분포 = 함수에 대한 믿음)   │
       └──────────────┬───────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                            │
        ▼                            ▼
   exploit μ(x)              explore σ(x)
        │                            │
        └─────────┬──────────────────┘
                  ▼
       a_UCB(x) = μ(x) + √β · σ(x)
                  │
                  │ Sobol 후보 + L-BFGS로 argmax
                  ▼
              x_next 결정
                  │
                  ▼
            f(x_next) 평가 → 데이터에 추가
                  │
                  └─── 반복 ───┐
                               ▼
                            예산 끝
```

---

## 짧은 예제 코드 (scikit-learn + scipy)

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats.qmc import Sobol

# 목적: f(x)를 최대화한다고 가정 (실제로는 비싼 블랙박스)
def f(x):
    return -(x - 2.0)**2 * np.sin(3*x) + 0.1*np.random.randn()

# --- 초기화 ---
lb, ub = 0.0, 5.0
rng = np.random.default_rng(0)
X = rng.uniform(lb, ub, size=(5, 1))
y = np.array([f(xi[0]) for xi in X])

kernel = C(1.0) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6)

# --- BO 루프 ---
beta = 4.0
for t in range(20):
    gp.fit(X, y)

    # Sobol 후보 (1D에선 사실 linspace로도 충분)
    sampler = Sobol(d=1, scramble=True, seed=t)
    X_cand = lb + (ub - lb) * sampler.random(n=1024)

    mu, sigma = gp.predict(X_cand, return_std=True)
    ucb = mu + np.sqrt(beta) * sigma
    x_next = X_cand[np.argmax(ucb)].reshape(1, -1)

    y_next = f(x_next[0, 0])
    X = np.vstack([X, x_next])
    y = np.append(y, y_next)

best_idx = np.argmax(y)
print("best x:", X[best_idx], "best y:", y[best_idx])
```

---

### 한 줄 요약

**베이지안 최적화** = "GP로 함수에 대한 사후분포(평균 μ + 불확실성 σ)를 만들고, **UCB = μ + √β·σ** 같은 acquisition을 **Sobol 후보 + 국소 최적화**로 최대화해서 다음 평가 점을 정하고, 다시 GP를 업데이트하는 과정을 반복."