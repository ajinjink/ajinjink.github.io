---
title: "[ML] Linear Regression"
date: 2026-05-09
categories: ["AI", "ML"]
tags: ["linear regression", "regression", "MLE", "Bernoulli distribution", "Gaussian distribution", "RSE", "RSS", "MSE", "feature selection", "best subset selection", "stepwise selection", "stagewise selection", "cross validation", "k-fold"]
use_math: true
---

## 선형 회귀(Linear Regression)

- 입력 변수(X)와 예측하고자 하는 출력 변수(Y) 사이에 선형적으로 비례하는 관계(기울기가 일정함)가 있다고 가정하는 가장 간단한 형태의 지도학습 모델
- 단순 선형 회귀:
  - 입력 변수가 하나일 때, 모델은 $y = \beta_0 + \beta_1 x + \epsilon$ 형태의 1차 함수 꼴
  - $\beta_0$ : 절편(X가 0일 때의 기본값)
  - $\beta_1$ : 기울기(X의 변화에 따른 Y의 변화량)를 의미하는 계수(Coefficient)

---

## 최소제곱법(Least Squares Method):

- i번째 x에 대한 예측 결과값을 $\hat y_i = \hat \beta_0 + \hat \beta_1 \textbf x_i$ 라고 할 때
- **오차(Error)** : 실제 관측값($y_i$)과 모델이 예측한 값($\hat{y_i}$)의 차이
- **RSS(Residual Sum of Squares)** : 이 오차들을 제곱하여 모두 더한 것
  - 완벽하게 예측했다면 0
  - $$RSS = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (y_i - \hat{y}*i)^2 = \sum*{i=1}^n (y_i - (\beta_0 + \beta_1 x_i))^2$$
- RSS를 최소화하는 $\beta_0$와 $\beta_1$을 찾기 위해 편미분하여 0이 되는 지점 탐색
  - $\frac{\partial RSS}{\partial \hat \beta_0} = 0 \implies \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$
  - <span>$$\frac{\partial RSS}{\partial \beta_1} = 0 \implies \hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$$</span>
    - $\bar{x}, \bar{y}$는 각각 $x$와 $y$의 평균
- 최소제곱법 : 오차를 최소화하기 위해 식을 미분하여 0이 되는 지점을 찾는 방식으로 최적의 $\beta_0$와 $\beta_1$ 값을 구하는 방법

---

### 모델 평가 지표 및 계수 해석

- 평가 지표:
  - **RSE (Residual Standard Error) / RMSE**: 평균적으로 데이터 포인트당 얼마나 틀렸는지
    - $RSE = \sqrt{\frac{RSS}{n-2}}$
    - $RMSE = \sqrt{\frac{RSS}{n}}$
  - **$R^2$ (R-squared):** 데이터가 가진 전체 분산 중 모델이 예측하지 못하고 틀린 분산을 제외한 비율
    - 모델이 데이터의 변동(분산)을 얼마나 잘 설명하는지
    - 0에서 1 사이의 값을 가지며, 1에 가까울수록 모델이 예측을 잘 수행했음을 의미
    - $R^2 = \frac{\text{TSS} - \text{RSS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}}$
    - **R² = 1** → 모델이 데이터 변동을 **완벽하게** 설명 (잔차 = 0)
    - **R² = 0.8** → 전체 변동의 **80%를 설명**, 20%는 설명 못 함
    - **R² = 0** → 모델이 아무것도 설명 못 함 (그냥 평균값 예측과 동일)
    - 단순 선형회귀(변수 1개)에서는 $$R^2 = r^2$$. 즉 **피어슨 상관계수 r의 제곱**과 같아짐.
      - x와 y의 선형 관계가 강할수록 R²도 높아짐
- 계수 해석 시 주의점:
  - **상관관계 != 인과관계:** 두 변수 간에 상관성이 있다고 해서 하나가 다른 하나의 원인이 된다고 단정 지을 수 없다
  - **스케일(Scale)의 중요성:** 계수($\beta$)의 크기가 크다고 해서 반드시 더 중요한 변수인 것은 아니다.
    - 입력 데이터의 단위(예: 기압은 1000 단위, 온도는 10 단위)에 따라 스케일 차이를 상쇄하기 위해 계수 크기가 조정될 수도.

---

### 왜 최소제곱법을 사용하는가?

**MLE, Maximum Likelihood Estimation** 개념을 통해 선형 회귀에서 오차의 제곱을 최소화

- **IID 가정:** 우리가 수집한 데이터는 서로 독립적(Independent)이고 동일한 확률 분포(Identically Distributed)에서 추출되었다고 가정
- **Likelihood** : 확률 분포의 파라미터 $\theta$가 주어졌을 때, 관측된 $n$개의 데이터($x_1, \dots, x_n$)가 나올 확률들의 곱
  - IID이기 때문에 각 샘플의 확률은 독립이라서 곱할 수 있음.
  - $$L(\theta) = \prod_{i=1}^n P(x_i | \theta)$$
- **Log-Likelihood** : 계산의 편의를 위해 곱셈을 덧셈으로 바꾸고자 로그를 씌운 것
  - 로그를 씌워도 최댓값을 갖는 지점은 변하지 않음 
  - $$\log L(\theta) = \sum_{i=1}^n \log P(x_i | \theta)$$
  - $$\underset{\beta}{\text{maximize}} \log L(\beta) \iff \underset{\beta}{\text{minimize}} \sum_{i=1}^{n}(y_i - \beta^T x_i)^2$$
- **MLE의 적용**
  - 선형 회귀에서 예측값($y$)은 입력변수들의 선형 결합($\beta^T X$)을 평균으로 하고, 입력값에 상관없이 일정한 분산($\sigma^2$)을 가지는 정규분포를 따른다고 가정
  - 관측된 데이터 전체가 이 정규분포에서 나왔을 확률을 최대화하는 파라미터 $\beta $를 추정
  - 계산
    - 함수의 **극값(최대/최소)**은 1차 도함수(미분)가 0이 되는 지점에서 발생. 기울기(Gradient)를 0으로 설정
    - <span>$$\nabla \ell(\theta) = 0 \quad \text{또는} \quad \frac{\partial \ell(\theta)}{\partial \theta_j} = 0$$</span>
    - 이 조건만으로는 **최대**인지 **최소**인지 구분 불가 → 추가 확인 필요
    - 2차 도함수를 담은 행렬인 **헤시안**을 확인
      - <span>$$[H(\theta)]_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}$$</span>
    - $H(\theta)$가 **음정치(negative-definite)**일 때 local maximum (극대)

---

예시 1
  - 베르누이 분포를 따른다고 가정할 때 
  - <span>$$p_\theta(\mathbf{x}) = \theta^\mathbf{x}(1-\theta)^{1-\mathbf{x}}$$</span>
    - $\mathbf{x} \in {0, 1}$ : 샘플 값
    - $\theta$ : 미지의 파라미터, $x_i = 1$이 될 확률
  - $\hat{\theta} = \underset{\theta}{\text{argmax}} \ L(\theta; \mathbf{x}_1, ..., \mathbf{x}_n)$
  - 로그 변환
    - $$\ell(\theta) = \underset{\theta}{\text{argmax}} \sum_{i=1}^{n} \log p_\theta(\mathbf{x})= \underset{\theta}{\text{argmax}} \sum_{i=1}^{n} \log \theta^{x_i}(1-\theta)^{1-x_i} = \log\theta \sum_{i=1}^{n} \mathbf{x}_i + \log(1-\theta)\sum_{i=1}^{n}(1-\mathbf{x}_i)$$
  - 미분하여 0으로 설정
    - $$\frac{\partial \ell(\theta)}{\partial \theta} = \frac{1}{\theta}\sum_{i=1}^{n}\mathbf{x}*i - \frac{1}{1-\theta}\sum*{i=1}^{n}(1-\mathbf{x}_i) = 0$$
  - 식 정리하면
    - $$\sum_{i=1}^{n}\mathbf{x}_i = n\theta$$
    - $$ \theta = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i$$
  - 한 번 더 미분해서 부호 봐. (-)가 나와야 극대 보장

---

예시 2
  - 데이터가 가우시안 분포를 따른다고 가정할 때 
  - <span>$$p_\theta(\mathbf{x}) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$</span>
    - 파라미터: $\theta = {\mu, \sigma}$ → **평균**과 **표준편차** 둘 다 추정해야 함
  - <span>$$\ell(\theta) = \sum_{i=1}^{n} \log \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x_i-\mu}{\sigma}\right)^2}$$</span>
  - 로그 변환
    - $$\ell(\theta) = \sum_{i=1}^{n} \log \frac{1}{\sigma\sqrt{2\pi}} - \sum_{i=1}^{n}\frac{1}{2}\left(\frac{\mathbf{x}_i - \mu}{\sigma}\right)^2$$
  - $\mu$에 대해 편미분
    - $$\frac{\partial \ell(\theta)}{\partial \mu} = \sum_{i=1}^{n}\left(\frac{\mathbf{x}_i - \mu}{\sigma^2}\right) = 0$$
    - $$\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i$$
  - $\sigma$에 대해 편미분
    - $$\frac{\partial \ell(\theta)}{\partial \sigma} = -\frac{n}{\sigma} - \frac{1}{2} \cdot (-2)\sigma^{-3}\sum_{i=1}^{n}(\mathbf{x}_i - \mu)^2 = 0$$
    - $$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \mu)^2$$
  - 한 번 더 미분해서 부호 봐. (-)가 나와야 극대 보장

---

## Linear Regression에서의 MLE

예측값 $y$는 선형 결합($\beta^T x$)을 평균으로 하고, 일정한 분산($\sigma^2$)을 가지는 정규분포를 따른다고 가정

<span>
$$\mathbf{y} = \boldsymbol{\beta}^\top \mathbf{x} + \varepsilon, \quad \text{where } \varepsilon \sim \mathcal{N}(0, \sigma^2)$$</span>

<span>$$\mathbf{y} \approx \boldsymbol{\beta}^\top \mathbf{x} = \beta_0 + \beta_1 x_1 + ... + \beta_d x_d$$</span>

<span>$$\mathbf{y} \mid \mathbf{x} \sim \mathcal{N}(\boldsymbol{\beta}^\top \mathbf{x},\ \sigma^2)$$</span>

- 평균: $\mathbb{E}(\mathbf{y} \mid \mathbf{x}) = \boldsymbol{\beta}^\top \mathbf{x}$
- 분산: $\sigma^2$ = 상수 (constant)



1. $p_\theta(\mathbf{x})$ 설정

    선형 회귀에서:

    - 추정할 파라미터: $\theta = \boldsymbol{\beta}$
    - 타겟 변수가 $\mathbf{y}$이므로, 평균이 $\boldsymbol{\beta}^\top\mathbf{x}$인 가우시안:
    - $$p_\theta(\mathbf{y} \mid \mathbf{x}) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{y - \boldsymbol{\beta}^\top \mathbf{x}}{\sigma}\right)^2}$$



2. 로그 변환
   $$\ell(\theta) = \sum_{i=1}^{n} \log p_\theta(\mathbf{y}_i \mid \mathbf{x}_i) = \sum_{i=1}^{n} \log \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{\mathbf{y}_i - \boldsymbol{\beta}^\top \mathbf{x}_i}{\sigma}\right)^2} = \sum_{i=1}^{n} \log \frac{1}{\sigma\sqrt{2\pi}} - \sum_{i=1}^{n} \frac{1}{2}\left(\frac{\mathbf{y}_i - \boldsymbol{\beta}^\top \mathbf{x}_i}{\sigma}\right)^2$$

3. 무관한 항 제거 (simplification)

    $$\underset{\theta}{\text{argmax}}\ \ell(\theta) = \underset{\boldsymbol{\beta}}{\text{argmax}} \left[ \underbrace{\sum_{i=1}^{n} \log \frac{1}{\sigma\sqrt{2\pi}}}_{\boldsymbol{\beta}\text{와 무관 → 제거}} - \underbrace{\sum_{i=1}^{n} \frac{1}{2}}_{\boldsymbol{\beta}\text{와 무관 → 제거}}\left(\frac{\mathbf{y}_i - \boldsymbol{\beta}^\top \mathbf{x}_i}{\sigma}\right)^2 \right]$$

    $$= \underset{\boldsymbol{\beta}}{\text{argmax}} \left[ -\sum_{i=1}^{n}(\mathbf{y}_i - \boldsymbol{\beta}^\top \mathbf{x}_i)^2 \right]$$

    $$\text{최대화 → 최소화 (부호 반전)} = \underset{\boldsymbol{\beta}}{\text{argmin}} \sum_{i=1}^{n}(\mathbf{y}_i - \boldsymbol{\beta}^\top \mathbf{x}_i)^2$$

4. <span>$$\therefore \text{Linear Regression에서의 MLE} \equiv \text{Least Square (최소제곱법)}$$</span>

---

## 행렬 표기법 및 정규 방정식 (Matrix Notation & Normal Equation)

여러 개의 변수를 다룰 때 식을 행렬을 이용해 간결하게 표현 가능

- n개의 regression equation을 matrix notation으로 표현
    - $$\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix} = \begin{bmatrix} 1 & x_{11} & \cdots & x_{1p} \\ 1 & x_{21} & \cdots & x_{2p} \\ \vdots & \vdots & \vdots & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{bmatrix} \begin{bmatrix} \beta_0 \\ \vdots \\ \beta_p \end{bmatrix} + \begin{bmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{bmatrix} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$
- Sum of squared residuals를 최소화

    - 목적함수 (Least Square)
        - $$\hat{\beta} = \underset{\beta}{\text{argmin}} \sum_{i=1}^{n}(\mathbf{y}_i - \beta^\top \mathbf{x}_i)^2$$
            - $\mathbf{y}_i$ : 스칼라 (1×1)
            - $\beta$ : 파라미터 벡터 (d×1)
            - $\mathbf{x}_i$ : 입력 벡터 (d×1)
    - 행렬 표기법으로 변환
        - $$\hat{\beta} = \underset{\beta}{\text{argmin}}\ (\mathbf{Y} - \mathbf{X}\beta)^\top(\mathbf{Y} - \mathbf{X}\beta)$$
        - $\mathbf{Y}$ : 타겟 벡터 ($n \times 1$)
        - $\mathbf{X}$ : 데이터 행렬 ($n \times d$)
        - $\beta$ : 파라미터 벡터 ($d \times 1$)
        - $(\mathbf{Y} - \mathbf{X}\beta)^\top(\mathbf{Y} - \mathbf{X}\beta)$ = 잔차 벡터의 내적 = $\sum$ 잔차$^2$

    - Normal Equation 도출 (미분 -> 0)

        - RSS($\beta$) = $(\mathbf{Y} - \mathbf{X}\beta)^\top(\mathbf{Y} - \mathbf{X}\beta)$ 를 $\beta$로 미분:

        - 미분
            - <span>$$\frac{\partial \text{RSS}(\beta)}{\partial \beta} = -\mathbf{X}^\top(\mathbf{Y} - \mathbf{X}\beta) - (\mathbf{Y} - \mathbf{X}\beta)^\top\mathbf{X}$$</span>
            - <span>$$= -\mathbf{X}^\top(\mathbf{Y} - \mathbf{X}\beta) - \mathbf{X}^\top(\mathbf{Y} - \mathbf{X}\beta) = -2\mathbf{X}^\top(\mathbf{Y} - \mathbf{X}\beta) = 0$$</span>
        - 전개
            - <span>$$\mathbf{X}^\top\mathbf{Y} - \mathbf{X}^\top\mathbf{X}\beta = 0$$</span>
            - <span>$$\hat{\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{Y} \quad \leftarrow \textbf{Normal Equation (정규 방정식)}$$</span>

        - 차원 : $(d \times d)^{-1}(d \times n)(n \times 1) = d \times 1$ 



### Bias Trick

- 원래 모델엔 절편 $\beta_0$가 따로 있어서 불편
- 행렬식의 편의를 위해 입력 변수 $x$ 벡터의 맨 앞에 무조건 1을 추가하여 절편 $\beta_0$를 행렬 내적 연산 하나로 통합
- $$x' = [1, x_1, x_2, \dots, x_d]^T, \quad \beta' = [\beta_0, \beta_1, \dots, \beta_d]^T \implies y = {\beta'}^T x'$$



### 통계적 특성 (Expectation & Variance)

- 기댓값 Unbiasedness

  - $\mathbb{E}[\hat{\beta} \mid \mathbf{X}] = \mathbb{E}[(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{Y} \mid \mathbf{X}]$

  - $\mathbb{E}[a\mathbf{X}] = a\mathbb{E}[\mathbf{X}] = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \mathbb{E}[\mathbf{Y}\mid\mathbf{X}] $

  - $\mathbf{Y}\mid\mathbf{X} \sim \mathcal{N}(\beta^\top\mathbf{X}, \sigma^2)$ 이므로 $\mathbb{E}[\mathbf{Y}\mid\mathbf{X}] = \mathbf{X}\beta$:

    <span>$$= (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top(\mathbf{X}\beta) = \underbrace{(\mathbf{X}^\top\mathbf{X})^{-1}(\mathbf{X}^\top\mathbf{X})}_{=\mathbf{I}}\beta$$</span>

  - <span>$$\mathbb{E}[\hat{\beta}\mid\mathbf{X}] = \beta \quad \leftarrow \textbf{Unbiased (불편 추정량)}$$</span>

- 분산

  - $\text{Var}[\mathbf{AX}] = \mathbf{A}\text{Var}[\mathbf{X}]\mathbf{A}^\top$

  - <span>$$\text{Var}[\hat{\beta}\mid\mathbf{X}] = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \cdot \underbrace{\text{Var}[\mathbf{Y}\mid\mathbf{X}]}_{\sigma^2\mathbf{I}} \cdot \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}$$</span>

    <span>$$= \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}\underbrace{\mathbf{X}^\top\mathbf{X}}_{=\mathbf{I} \text{로 소거}}(\mathbf{X}^\top\mathbf{X})^{-1}$$</span>

  - <span>$$\text{Var}[\hat{\beta}\mid\mathbf{X}] = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$$</span>
  - 데이터 $n$이 많아질수록 $(\mathbf{X}^\top\mathbf{X})^{-1}$이 작아짐 → **분산 감소** → **추정이 더 안정적**





---

## Bias-Variance Decomposition

줄이고자 하는 모델의 총 오차(MSE, 평균 제곱 오차)는 바이어스의 제곱과 분산의 합으로 분해될 수 있음

- MSE 분해 수식:

  <span>$$\text{MSE}(\hat{\theta}) = \mathbb{E}[\|\hat{\theta} - \theta\|^2] = \mathbb{E}\left[\sum_{j=1}^{d}(\hat{\theta}_j - \theta_j)^2\right]$$</span>

  - 추정값과 실제값의 **평균 제곱 차이** = 총 오차
  - **편향(Bias):** $Bias(\hat \theta) = E[\hat{\theta}] - \theta$
    - 추정값의 **평균**이 실제 참값 $\theta $에서 **얼마나 벗어났는지**
    - Bias=0 이면 **Unbiased (불편 추정량)**
  - **분산(Variance):** $Var(\hat \theta) = E[(\hat{\theta} - E[\hat{\theta}])^2]$
    - 추정값이 **자기 자신의 평균으로부터 얼마나 퍼져있는지**
    - 데이터 샘플링에 따른 예측값들의 변동성
  - 증명
    - <span>$$\text{MSE}(\hat{\theta}) = \text{tr}(\text{Var}(\hat{\theta})) + |\text{Bias}(\hat{\theta})|^2$$</span>
      - $\mathbb{E}[(\hat{\theta} - \theta)^2] = \mathbb{E}[(\hat{\theta} - \mathbb{E}(\hat{\theta}))^2] + (\mathbb{E}(\hat{\theta}) - \theta)^2$
      - 이거 전개하면 $$\mathbb{E}[(\hat{\theta} - \theta)^2] = \underbrace{\mathbb{E}[(\hat{\theta} - \mathbb{E}(\hat{\theta}))^2]}_{\text{Variance}} + \underbrace{(\mathbb{E}(\hat{\theta}) - \theta)^2}_{\text{Bias}^2}$$



선형 회귀에 적용

- <span>$$\text{MSE}(\hat{\theta} \mid \mathbf{X}) = \text{Var}(\hat{\theta} \mid \mathbf{X}) + \text{Bias}(\hat{\theta} \mid \mathbf{X})^2$$</span>

- <span>$$\text{MSE}(\hat{\theta} \mid \mathbf{X}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1} + 0 = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$$</span>

- 데이터 $n \to \infty$ 이면 $(\mathbf{X}^\top\mathbf{X})^{-1} \to 0$ → **MSE도 0으로 수렴**

- Gauss-Markov Theorem

  - **MLE는 최량 불편 추정량(BLUE: Best Linear Unbiased Estimator)이다.** 즉, Unbiased한 모든 추정량 중에서 MLE의 분산이 가장 작다.
  - <span>$$\text{어떤 unbiased 추정량 } \tilde{\theta} \text{ 에 대해서도: } \text{Var}(\hat{\theta}_{\text{MLE}}) \leq \text{Var}(\tilde{\theta})$$</span>

- Bias-Variance Tradeoff

  - $$\text{MSE} = \underbrace{\text{Variance}}_{\text{줄이면}} + \underbrace{\text{Bias}^2}_{\text{늘어남}}$$

    |                  | Bias   | Variance | 결과         |
    | ---------------- | ------ | -------- | ------------ |
    | 모델이 너무 단순 | 높음 ↑ | 낮음 ↓   | Underfitting |
    | 모델이 너무 복잡 | 낮음 ↓ | 높음 ↑   | Overfitting  |
    | 적절한 모델      | 적당   | 적당     | Best       |


---

## Linear Regression with Categorical Features

### Categorical (Qualitative) Predictors

- 어떤 predictor들은 quantitative가 아니라 qualitative. discrete한 값들을 가짐
  - Categorical predictors 또는 **factor variables**라고도 불림
  - 예시: credit card data
    - 7개의 quantitative variable 외에, 4개의 qualitative variable이 있음
    - `own` (집 소유 여부), `student` (학생 여부), `status` (혼인 여부), `region` (East, West, South)


- 예시: 집 소유 여부(`own`, $x_i$)에 따른 credit card balance(`y`) 차이 조사
    - Binary variable로 표현:
    - $$x_i = \begin{cases} 1 & \text{if } i\text{th person owns a house} \\ 0 & \text{if } i\text{th person does not own a house} \end{cases}$$

- 모델:
  - $$y_i = \beta_0 + \beta_1 x_i + \epsilon_i = \begin{cases} \beta_0 + \beta_1 + \epsilon_i & \text{if } i\text{th person owns a house} \\ \beta_0 + \epsilon_i & \text{if } i\text{th person does not own a house} \end{cases}$$
    - input : 0 or 1
    - 집이 없는 사람 대비 집이 있는 사람이 돈을 추가로 얼마나 더 빌리는가($\beta_1$)
    - $\beta_1$가 (-)일 수도. 집이 있는 사람이 덜 빌린다

#### Credit Data 결과
  - Non-owner의 평균 credit card debt: $509.80
  - Owner는 추가로 19.73의 debt → 총 509.80 + $19.73\ (\beta_1) = $529.53

|               | Coefficient |
| ------------- | ----------- |
| Intercept     | 509.80      |
| Owner | 19.73      |

---

### 두 개 이상의 Level을 가지는 Qualitative Predictor

Level이 2개보다 더 클 때는 추가적인 variable이 필요

- 예: `region` (East, West, South)에 대해 두 개의 dummy variable 생성:

  - $$x_{i1} = \begin{cases} 1 & \text{if } i\text{th person is from the South} \\ 0 & \text{otherwise} \end{cases}$$

  - $$x_{i2} = \begin{cases} 1 & \text{if } i\text{th person is from the West} \\ 0 & \text{otherwise} \end{cases}$$
  - 두 변수가 모두 0이면 East에 산다고 가정

- **모델**:
  - $$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \epsilon_i = \begin{cases} \beta_0 + \beta_1 + \epsilon_i & \text{if from the South} \\ \beta_0 + \beta_2 + \epsilon_i & \text{if from the West} \\ \beta_0 + \epsilon_i & \text{if from the East} \end{cases}$$
  - East를 기준으로 South, West가 얼마나 더 쓰는지 차이를 학습

- Dummy variable의 개수는 level 수보다 **하나 적게** 가능
- Dummy variable이 없는 level (여기서는 East)을 **baseline**이라 함.

  

#### Credit Data 예시

|               | Coefficient |
| ------------- | ----------- |
| Intercept     | 531.00      |
| region[South] | −18.69      |
| region[West]  | −12.50      |

- Baseline인 East의 예측 balance는 $531.00
- South 사람은 East보다 $18.69 적은 debt
- West 사람은 East보다 $12.50 적은 debt

---

### One-hot Encoding

- categorical variable을 표현할 때 굳이 baseline를 두지 않고, 모든 level에 변수 사용

  - 예: color = {red, blue, green}
    - $x_1$ : red, $x_2$ : blue, $x_3$ : green

  - 오직 하나만 1, 나머지는 모두 0
  - color = red 이면 $x_1 = 1, x_2 = 0, x_3 = 0$

- 다차원 벡터에서 한 개만 1, 나머지는 다 0

------

## Modeling Interactions

### Interactions

- **Additive assumption 제거**: interaction과 nonlinearity 도입

- $$\text{Sales} = \beta_0 + \beta_1 \cdot \text{TV} + \beta_2 \cdot \text{Radio} + \beta_3 \cdot \text{Newspaper}$$

  - TV 광고, Radio 광고, Newspaper 광고를 어떻게 분배해야 Sales가 최대가 될까
  - 이 linear regression 수식에서는 한 광고 매체의 sales에 대한 효과는 다른 매체의 지출량과 무관하다고 가정
  - TV가 1단위 증가할 때 sales 평균 효과가 radio 지출량과 무관하게 항상 $\beta_1$임을 의미

#### Synergy/Interaction Effect

- 만약에 서로 다른 매체의 광고가 서로 영향을 준다면? 시너지가 나거나 악영향이 있다면?
- 마케팅에서는 **synergy effect**, 통계에서는 **interaction effect**라 부름
- TV 또는 radio 중 하나만 낮으면, 실제 sales는 linear model 예측보다 낮음
- 광고가 두 매체에 나뉘어 있으면, 모델은 synergy를 무시하므로 sales를 **과소추정**

---

### Interaction Term Modeling

- 만약 radio 광고 지출이 TV 광고 효과를 증가시킨다면, TV의 slope이 radio가 증가할수록 함께 증가해야 함

- 모델 형태:

  - $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \boxed{\beta_3 x_1 x_2} + \epsilon$$

  - $$\text{Sales} = \beta_0 + \beta_1 \cdot \text{TV} + \beta_2 \cdot \text{Radio} + \underbrace{\beta_3 \cdot (\text{TV} \cdot \text{Radio})}_{\text{Interaction term}} + \epsilon$$

#### 결과 분석

|            | Coefficient |
| ---------- | ----------- |
| Intercept  | 6.7502      |
| TV         | 0.0191      |
| radio      | 0.0289      |
| TV × radio | 0.0011      |

- TV x radio 에 대한 coefficient가 0.0011 이라서 상관관계가 없다 X
  - TV x radio 값이 커서 그럴 수도
  - 시너지 효과만 표시
- Interaction model의 $R^2$는 96.8% <-> interaction term 없는 모델은 89.7%
  - interaction 있다
- $(96.8 - 89.7)/(100 - 89.7) = 69%$ : additive model 이후 남은 sales 변동성 중 69%가 interaction term으로 설명됨
- TV 광고 1,000 증가 시 sales 증가량: $(\hat\beta_1 + \hat\beta_3 \times \text{radio}) \times 1000 = 19.1 + 1.1 \times \text{radio}$ units
- Radio 광고 1,000 증가 시 sales 증가량: $(\hat\beta_2 + \hat\beta_3 \times \text{TV}) \times 1000 = 28.9 + 1.1 \times \text{TV}$ units

<br>

- 때때로 interaction term은 매우 작은 p-value를 가지지만, 관련된 main effect (TV, radio)는 그렇지 않을 수 있음
- **Hierarchy principle** : 모델에 interaction을 포함하면, 관련 main effect도 반드시 포함해야 함
  - 예: TV × radio를 포함하면 TV와 radio도 포함해야 함
  - 만약에 3개 interaction term을 넣을 거면, 모든 2개 interaction term, 모든 1개 term도 모두 있어야 함
  - 모델에 main effect term이 없다면 interaction term이 main effect까지 포함하게 됨
    - 각각 개별 요소에 대한 단일 영향을 계산할 수 없음

---

### Nonlinear Relationships

- 두 개의 다른 variable 간 interaction뿐만 아니라, 같은 variable을 여러 번 곱할 수도 있음
- 예시: Auto data set (mpg vs horsepower)
  - input : 마력
  - output : 연비
  - **Yellow**: horsepower만 사용한 simple linear regression
  - **Blue**: horsepower²를 포함한 linear regression
  - **Green**: horsepower의 5차 다항식까지 포함한 linear regression

#### Nonlinear을 포함하는 방법

- Linear model에 non-linear association을 도입하는 간단한 방법: predictor의 **변환된 버전**을 포함

- <span>$$\text{mpg} = \beta_0 + \beta_1 \times \text{horsepower} + \beta_2 \times \text{horsepower}^2 + \epsilon$$</span>
  - 이것은 linear model인가? O
  - $X_1 = \text{horsepower}$, $X_2 = \text{horsepower}^2$로 보면 multiple linear regression임

- 고차항을 넣을 때는 반드시 저차항도 넣어야 함

|             | Coefficient |
| ----------- | ----------- |
| Intercept   | 56.9001     |
| horsepower  | −0.4662     |
| horsepower² | 0.0012      |

------

## Feature Selection

### Least Square Solution은 항상 존재하는가?

- Least square estimator: $\hat{\boldsymbol{\beta}} = [\hat\beta_0, \hat\beta_1, ..., \hat\beta_p]^\top = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$
- $\mathbf{X}^\top\mathbf{X}$가 invertible하지 않으면 계산 불가능
  - $\mathbf{X}^\top\mathbf{X}$는 두 column이 같으면 invertible 하지 않음 (determinant가 0이 되면 역행렬 존재 X. $\mathbf{X}$가 whole rank가 아닐 때)
  - $\mathbf{X}^\top\mathbf{X} \in \mathbb{R}^{p \times p}$, 각 column은 input feature $\mathbf{X}$의 해당 dimension을 의미
  - 두 feature가 중복(duplicate)된 상태



- feature가 완벽히 correlated 되진 않았지만 strongly correlated되어 있는 경우 (똑같지는 않아서 수학적으로 역행렬은 존재하는데, 칼럼 하나를 알면 다른 칼럼을 거의 맞출 수 있는 경우)에도 regression 수식이 제대로 안 나옴


### Feature Selection이 필요한 이유

- (거의) duplicate한 feature의 risk 제거
- subset으로 학습 시켰을 때 더 예측을 잘 할 수도 (Noisy feature에서 오는 unreliable prediction의 risk 감소)
- 있으나 없으나 비슷할 수도
- 더 적은 feature로 모델을 lighter and more efficient하게 만들기

---

### Best Subset Selection

가능한 모든 subset에 대해 least squares fit을 계산하고, training error와 model size 사이의 균형을 잡는 기준으로 선택.

1. $M_0$를 **null model**로 표기 (데이터 칼럼 사용 X, sample mean 예측)
2. $k = 1, 2, ..., p$ 에 대해:
   - 정확히 $k$개의 predictor를 포함하는 모든 $\binom{p}{k} = \frac{p!}{k!(p-k)!}$개 모델을 fit
   - 이 $\binom{p}{k}$개 모델 중 best를 선택하고 $M_k$로 표기 (변수를 k개 써서 만들 수 있는 가장 좋은 모델)
     - Best는 가장 작은 RSS, 동등하게 가장 큰 $R^2$
3. $M_0, ..., M_p$ 중 어떤 metric (예: prediction error, $R^2$) 에 따라 single best model 선택

#### 특징

- 각 step에서 **전체 coefficient가 re-optimize됨**
  - 칼럼 1개 쓰는 모델, 2개 쓰는 모델, 3개 쓰는 모델, ... 찾을 때마다 { 사용하는 칼럼, 각 칼럼에 대한 coefficient }가 다 바뀜
  - additive하게 { 칼럼, coefficient }가 하나씩 늘어나는 게 아니야 
- 2개 variable의 best 조합이 이전 step에서 선택된 variable을 반드시 포함하지는 않음 (예: $x_2$가 1-variable에서 선택되었다고 2-variable에 $x_2$가 들어간다는 보장 없음)

#### Best Subset Selection의 문제점

- 계산 비용
  : $2^p$개의 모델을 검토해야 함
  - $p = 40$이면 10억 개 이상의 모델

- 통계적 문제 (when $p$ is large)
  : 큰 search space → training data에서만 잘 보이는 모델을 찾을 확률 증가 (future data에는 예측력 없음)
  - **Curse of dimensionality**(차원의 저주)라고 부름 - 지수적으로 더 많은 데이터가 있어야 같은 예측 성능을 낼 수 있다
  - **Overfitting**과 coefficient estimate의 **high variance** 초래

- 이런 이유로, 더 제한된 모델 집합을 탐색하는 **stepwise methods**가 매력적인 대안

---

### (Forward) Stepwise Selection

Predictor 없는 모델부터 시작해, 한 번에 하나씩 predictor를 추가 (모든 predictor가 들어갈 때까지)

- 각 step에서, fit을 가장 크게 개선하는 variable을 추가

1. $M_0$: null model
2. $k = 0, 1, ..., p-1$에 대해:
   - $M_k$에 1개의 predictor를 추가한 $p - k$개 모델을 고려
   - 그 $p - k$개 (남아있는 predictors) 중 best를 선택하여 $M_{k+1}$로 표기
3. $M_0, ..., M_p$ 중 metric에 따라 single best model 선택

#### 특징

- **이전 step에서 선택된 variable은 변경되지 않음**
  - 처음에 $M_0$은 아무 칼럼 사용하지 않고 그냥 평균 표시
  - $M_1$에서 $x_2$를 사용을 했어
  - $M_2$에서는 $x_2$를 사용하고 있는 상태에서, 칼럼을 하나 더 추가한다면 어느 게 가장 예측 잘하는지 찾아서 추가
  - $M_3$에서도 $x_2$, $x_5$는 그대로 두고, 이 상태에서 칼럼을 하나 더 추가한다면 어느 게 가장 예측 잘하는지 찾아서 추가 
- 새 variable이 추가될 때 **이전에 fit된 coefficient들은 re-optimize됨** (variable 집합은 고정되지만 coefficient는 함께 다시 fit)

#### Best Subset vs Forward Stepwise 비교

- 계산 복잡도
  - Stepwise selection: $O(p^2)$
  - Best subset selection: $O(2^p)$
- Stepwise selection은 $2^p$개 모든 subset 중 best 모델을 찾는 것이 **보장되지 않음**

##### Credit data 예시

| # Variables | Best subset                           | Forward stepwise                   |
| ----------- | ------------------------------------- | ---------------------------------- |
| One         | rating                                | rating                             |
| Two         | rating, **income**                    | rating, **income**                 |
| Three       | rating, income, **student**           | rating, income, **student**        |
| Four        | **cards**, income, student, **limit** | rating, income, student, **limit** |

- Best subset: 어떤 4개 variable이든 자유롭게 선택 가능
- Stepwise: 이전 step에서 선택된 3개 variable이 반드시 포함되어야 함 (1개만 추가 선택)
  - 말하자면 greedy algorithm임

---

### (Backward) Stepwise Selection

모든 $p$개 predictor를 포함하는 full model에서 시작하여, 가장 쓸모없는 predictor를 하나씩 제거.

1. $M_p$: 모든 $p$개 predictor를 포함하는 full model
2. $k = p, p-1, ..., 1$에 대해:
   - $M_k$의 predictor 중 하나를 제외한 $k$개 모델을 고려 (총 $k-1$개 predictor)
   - 그 $k$개 모델 중 (어느 predictor를 제외하는 것이 가장 성능이 좋을지) best를 선택하여 $M_{k-1}$로 표기
3. $M_0, ..., M_p$ 중 metric에 따라 single best model 선택


#### 특징
- Forward stepwise, backward selection 모두 검토해야 하는 모델 개수는 똑같음
  - $1 + p(p+1)/2$개 모델 검토 → $p$가 클 때도 적용 가능
  - Forward, backward 모두 $O(p^2)$ 시간 복잡도
- Forward와 마찬가지로 backward stepwise도 best model을 보장하지 않음
- Backward selection은 sample 수 $n$이 variable 수 $p$보다 커야 함 (full model을 fit해야 하므로)
  - Forward stepwise는 $n < p$일 때도 사용 가능 → $p$가 매우 클 때 유일하게 사용 가능한 subset method
  - 실제로는 보통 $n \gg p$이므로 크게 걱정할 필요 없음

---

### (Forward) Stagewise Selection

이전 stage에서 fit된 coefficient를 **변경하지 않고** predictor를 한 번에 하나씩 추가.

- **같은 variable이 여러 번 선택될 수 있음** (이전에 fit된 coefficient를 조정하기 위함)

1. $M_0$: null model
2. $k = 1, 2, ..., K$ (몇 번 돌 건지 hyperparameter)에 대해:
   - $j = 1, 2, ..., p$에 대해:
     - $M_{k-1} + \beta x_j$에 대해 least squares fit, $M_k^{(j)}$로 표기
     - $p$개 모델 $M_k^{(1)}, ..., M_k^{(p)}$ 중 best를 선택하고 $M_k$로 표기
3. $M_0, ..., M_K$ 중 single best model 선택

#### Stagewise 진행 예시

- $y = 0.8$ → fit
- $y = 0.8 - 1.2x_2$ → fit (이전 0.8은 그대로 유지)
- $y = 0.8 - 1.2x_2 + 0.7x_5$ → fit
- $y = 0.8 - 1.2x_2 + 0.7x_5 + 0.3x_2 = 0.8 - 0.9x_2 + 0.7x_5$ → 같은 variable이 다시 선택되어 coefficient 보정

#### Stepwise vs Stagewise 비교

- Stepwise selection

  : 각 step에서 새 variable을 선택한 후, 지금까지 포함된 모든 variable에 대해 least square fit

  - **Greedy** 알고리즘: 한 번 추가된 variable은 제외될 수 없으나, 각 variable의 **coefficient는 변경 가능**

- Stagewise selection

  : 각 step에서 새 variable을 선택한 후, 새 variable에 대해서만 least square fit (이전 coefficient는 freezing)

  - **더 greedy**: 추가/fit된 coefficient도 변경 불가
  - 보정을 가능케 하기 위해 **같은 variable을 여러 번 추가** 허용

------

## 최적 모델 선택


- 가장 좋은 모델 : 가장 작은 RSS, 가장 큰 $R^2$, ...
- linear regression에서는 모든 predictor를 포함한 모델이 항상 가장 작은 RSS와 가장 큰 $R^2$를 가짐
  - 이런 의미에선 variable을 더 추가해도 잃을 게 없음
  - 어차피 추가된 variable이 쓸 데 없으면 계수가 0이 돼서 무시되겠지
- 우리는 **training error**가 아닌 **test error**가 낮은 모델을 원함
  - predictor 최적 개수 찾고자 하는데, test set에 대해서 결과가 잘 나오도록 학습시키는 건 cheating

- 일부 데이터를 test set으로 두고, 학습에는 사용 X
- 분할 방법
  - uniformly randomly, 10~20%
  - timestamp 기반 분할 (앞 4년으로 학습, 마지막 1년으로 평가)

<br>

- 많은 ML 모델에서 hyperparameter 설정 필요:
  - **Model parameters** (예: 선택할 predictor 수 $p$)
  - **Learning parameters** (예: learning rate)
- Model training과 같은 방식으로 반복: 여러 hyperparam 조합으로 모델을 학습하고 평가해 최선을 선택
- 이를 위해 training 시 **unseen examples**가 필요 (training과 같은 이유로)
- **Test data는 actual testing 외에는 절대 사용 금지**
- → Training data의 또 다른 일부를 **validation set**으로 설정해 hyperparameter 선택

---

### Cross Validation

1. **Step 1**: Training set으로 모델 학습
2. **Step 2**: Validation set으로 모델 평가
3. **Step 3**: 다른 hyperparameter로 Step 1-2 반복, best model 선택
4. **Step 4**: Step 3에서 선택된 hyperparam으로 **training + validation set**에서 모델 학습
5. **Step 5**: Step 4 모델을 eval set에서 평가

- Eval set은 Step 4를 끝낼 때까지 절대 사용 X
- 일반적 분할 예시: Training (70%) | Validation (20%) | Eval (10%)

---

### K-fold Cross Validation

- 데이터를 $K$개의 equal-sized block으로 분할
- $K$번 training 반복: 한 block을 제외한 모든 block에서 학습, 제외된 block에서 평가
- $K$개의 score를 평균
- validation은 training set 중 일부를 사용

| Iteration | Fold 1   | Fold 2   | Fold 3   | Fold 4   | Fold 5   |
| --------- | -------- | -------- | -------- | -------- | -------- |
| 1         | **Eval** | Train    | Train    | Train    | Train    |
| 2         | Train    | **Eval** | Train    | Train    | Train    |
| 3         | Train    | Train    | **Eval** | Train    | Train    |
| 4         | Train    | Train    | Train    | **Eval** | Train    |
| 5         | Train    | Train    | Train    | Train    | **Eval** |