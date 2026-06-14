---
title: "[RL] Finite Markov Decision Process (유한 마르코프 결정 과정)"
date: 2026-05-31
categories: ["AI", "RL"]
tags: ["MDP", "dynamics", "Markov", "Markov property", "absorbing state", "episodic task", "continuing task", "policy", "value function", "state value function", "action value function", "Bellman equation", "optimal policy", "optimal value function", "Bellman optimality"]
use_math: true
---


강화학습은 한 줄로 말하면 "시행착오를 통해 좋은 의사결정을 배우는" 문제.  
그런데 이걸 수학적으로 다루려면, "의사결정"이라는 추상적인 개념을 수식으로 표현할 수 있어야 함.  
여기서 등장하는 게 Markov Decision Process (MDP). 

MDP는 의사결정 상황을 다음 네 가지로 추상화한 framework:
- 상태 (state) — 지금 내가 어떤 상황에 놓여 있는가
- 행동 (action) — 내가 어떤 선택을 할 수 있는가
- 보상 (reward) — 그 선택이 얼마나 좋았는가
- 전이 (transition) — 그 선택의 결과로 어떤 새 상황이 펼쳐지는가


finite MDP : 강화학습 문제를 수학적으로 정교하게 다루기 위한 표준 framework
- 상태, 행동, 보상이 모두 유한 집합인 MDP
- 무한 차원을 다루지 않으니 수학이 깔끔해지고, 강화학습 이론의 출발점이 됨
- 평가적 피드백뿐 아니라 "서로 다른 상황에서 서로 다른 행동을 선택"하는 연관적 측면까지 포함하는, 순차적 의사결정의 고전적 형식화

---


## Agent-Environment Interface

MDP는 목표를 이루기 위해 상호작용으로부터 학습하는 문제를 단순한 frame으로 추상화함.

- **Agent**: 학습자이자 의사결정자
- **Environment**: agent 바깥의 모든 것

- discrete time step $t = 0, 1, 2, \dots$ 마다 agent는 상태 $S_t \in \mathcal{S}$ 를 받고, 행동 $A_t \in \mathcal{A}(s)$ 를 선택함. 
- 다음 단계에서 환경은 보상 $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$ 과 다음 상태 $S_{t+1}$ 을 돌려줌.

<pre style="font-family: monospace; line-height: 1.5;">
                 Action $A_t$
          ┌─────────────────────────┐
          │                         ▼
       [Agent]                   [Environment]
          ▲
          └─────────────────────────┘
          State $S_{t+1}$, Reward $R_{t+1}$
</pre>

<br>

한 스텝에서 무슨 일이 일어나는가:
- Agent가 현재 상태 $S_t$를 본다
- 그 상태를 기반으로 행동 $A_t$를 선택한다
- Environment가 반응해서, 새로운 상태 $S_{t+1}$과 보상 $R_{t+1}$을 돌려준다
- Agent는 그걸 받고 다시 1번부터 반복

이 과정을 시간순으로 나열하면 다음과 같은 trajectory 가 만들어짐:

$$
S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots
$$
- $S_0$ 에서 $A_0$ 를 했더니 보상 $R_1$ 을 받고 새 상태 $S_1$ 로 넘어감. 
- 거기서 $A_1$ 을 했더니 $R_2$ 와 $S_2$ 가 나옴. 계속 반복.
- 시간 $t$ 에서 행동 $A_t$ 를 했는데, 왜 보상은 $R_t$ 가 아니고 $R_{t+1}$ 일까?
    - 행동의 결과로 받는 보상
    - $A_t$ 를 했기 때문에 다음 스텝에서 $R_{t+1}$ 과 $S_{t+1}$ 이 *한꺼번에* 결정된다

<br>

청소 로봇을 예로 들면
- **상태**: 로봇의 배터리 잔량 (high, low)
- **행동**: search(캔 찾기), wait(대기), recharge(충전)
- **보상**: 캔을 모으면 +1, 배터리 방전되면 -3

일 때, trajectory는 :
<pre style="font-family: monospace; line-height: 1.5;">
S_0 = high  →  A_0 = search  →  R_1 = +1, S_1 = high
            →  A_1 = search  →  R_2 = +1, S_2 = low
            →  A_2 = wait    →  R_3 = 0,  S_3 = low
            →  A_3 = recharge → R_4 = 0, S_4 = high
            ...
</pre>


---

### Dynamics 함수



이제 환경이 "어떻게 반응하는가"를 수식으로 적어야 함. 

핵심 관찰: 직전 상태 $S_{t-1}$ 과 직전 행동 $A_{t-1}$ 이 주어지면, 그 다음에 무엇이 일어날지(=새 상태 $S_t$ 와 보상 $R_t$ )는 **확률 분포** 로 표현됨.

이걸 한 함수로 묶은 게 **dynamics 함수** $p$:

$$
p(s', r \mid s, a) \doteq \Pr\{S_t = s',\, R_t = r \mid S_{t-1} = s,\, A_{t-1} = a\}
$$
- 상태가 $s$ 이고 행동 $a$ 를 했을 때, 다음 상태가 $s'$ 이 되고 보상이 $r$ 이 될 확률


이 함수는 모든 $s, a$ 에 대해 다음을 만족함 :

$$
\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) = 1
$$
- 확률 분포 합 1
- 어떤 상태-행동 $(s, a)$ 를 고정해놓고, 가능한 모든 (다음 상태, 보상) 조합에 대해 확률을 다 더하면 1

---

### Markov property

- 위 정의에서 다음 상태와 보상의 분포가 오직 $(S_{t-1}, A_{t-1})$ 에만 의존함. 
- 더 옛날의 $S_{t-2}, A_{t-2}, \dots$ 같은 건 신경 안 씀.
- 상태 $S_t$ 가 필요한 모든 역사 정보를 이미 담고 있다
<br>
- 상태는 과거의 모든 agent-environment 상호작용에 대한 정보를 포함해야 함. 
- 이 조건을 만족하는 상태가 **Markov property** 를 가진다고 말함.

---

### Dynamics에서 파생되는 양들

dynamics $p(s', r \mid s, a)$ 로부터 다른 유용한 양들을 모두 계산할 수 있음.

**State-transition probability**:

보상은 신경 안 쓰고, 그냥 다음 상태가 $s'$ 일 확률만 알고 싶다

$$
p(s' \mid s, a) \doteq \Pr\{S_t = s' \mid S_{t-1} = s,\, A_{t-1} = a\} = \sum_{r \in \mathcal{R}} p(s', r \mid s, a)
$$
- 보상이 무엇이든 상관없으니까, 가능한 모든 보상값 $r$ 에 대해 확률을 더해버리면 됨
    - marginalization

**Expected reward (state-action)**:

상태 $s$ 에서 행동 $a$ 를 했을 때, 보상의 기댓값

$$
r(s, a) \doteq \mathbb{E}[R_t \mid S_{t-1} = s,\, A_{t-1} = a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \mid s, a)
$$
- 내부 합 $\sum_{s'} p(s', r \mid s, a)$: 다음 상태 $s'$ 가 무엇이든 상관없이 "보상이 $r$ 일 확률"을 계산 (marginalization)
- 외부 합 $\sum_r r \cdot \Pr(R_t = r \mid s, a)$: 기댓값 정의

**Expected reward (state-action-next state)**:

$s$ 에서 $a$ 를 했고, 그 결과 $s'$ 로 이동한 경우의 보상의 기댓값

$$
r(s, a, s') \doteq \mathbb{E}[R_t \mid S_{t-1} = s,\, A_{t-1} = a,\, S_t = s'] = \sum_{r \in \mathcal{R}} r \cdot \frac{p(s', r \mid s, a)}{p(s' \mid s, a)}
$$

---

### Agent-environment 경계

직관과 달리 agent와 environment의 경계는 물리적 경계와 다름.  
일반적인 규칙은 "agent에 의해 임의로 변경될 수 없는 모든 것은 환경의 일부"
- 로봇의 모터, 기계 부품, 센서 → environment의 일부 (물리 법칙에 따라 동작하지, agent가 마음대로 바꿀 수 없음)
- 보상을 계산하는 로직 → environment의 일부 (agent가 자기한테 보상을 마음대로 줄 수 있으면 학습이 무의미해짐)
- agent가 "선택할 수 있는 것"만 → **진짜 agent**

agent가 환경에 대해 아무것도 모른다는 뜻이 아니라, agent가 *임의로 바꿀 수 있는 영역의 한계*를 나타낸다는 뜻.

---

## Goals and Rewards

보상 $R_t \in \mathbb{R}$ 이 매 스텝마다 환경에서 agent로 흘러감.
그런데 강화학습은 이 단순한 숫자 하나로 agent의 *모든* 목표를 표현함.  
이걸 reward hypothesis 라고 부름:
- **Reward hypothesis**: 목표와 목적의 의미는, 오로지 누적 보상(cumulative reward)의 기댓값을 최대화하는 것으로 생각할 수 있다.

- 로봇이 걷기를 배우게 하고 싶다 → 앞으로 나아간 거리에 비례해서 보상
- 미로 탈출을 배우게 하고 싶다 → 매 스텝마다 -1 보상 (빨리 탈출할수록 손해가 적음)
- 체스 → 이기면 +1, 지면 -1, 비기면 0
- 청소 로봇 → 캔 하나 모으면 +1, 배터리 방전되면 -3

보상은 **무엇을** 이루어야 하는지를 알려주는 신호이지, **어떻게** 이루어야 하는지를 알려주는 신호가 아님.
- "이기면 +1" → ✅
- "상대 말을 잡을 때마다 +0.1, 중앙을 점령하면 +0.05" → ❌ 위험
    - agent는 이기는 법이 아니라 "상대 말을 많이 잡는 법"을 배워버림

---

## Returns and Episodes

agent의 목표는 장기적으로 보상 총합을 최대화하는 것.  
그런데 매 스텝 보상이 $R_1, R_2, R_3, \dots$ 처럼 *나열* 로 나오니까, "총합"이 정확히 뭔지 정의해야 함.

$$
G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \dots + R_T
$$
- $G_t$: 시점 $t$ 에서의 **return**. 지금부터 끝까지 받을 보상의 총합.
- $T$: 최종 시점

### Episodic vs Continuing tasks

만약에 T가 무한대라면?

#### Episodic task
- 게임, 미로 탈출 등 자연스럽게 부분(episode)으로 나뉘는 경우. 
- 각 episode는 **terminal state** 로 끝남.
- 끝나면 환경이 리셋되고 새 episode 시작. 
- 이때 $T$ 는 유한한 값이므로 위 식이 잘 정의됨.

- 비종단 상태 집합 $\mathcal{S}$ 와 종단 상태를 포함한 $\mathcal{S}^{+}$ 를 구분하기도 함.

#### Continuing task
- 자연스러운 종료 시점이 없는 경우 (24시간 돌아가는 공정 제어 시스템, 평생 학습하는 로봇)
- 위 정의대로면 $T = \infty$ 이고 합이 발산할 수 있음.
    - 매 스텝 +1만 받아도 합이 무한이 됨.
    - "최대화"라는 게 의미가 없어짐.

### Discounted return : 발산 문제 해결

두 경우를 통합하기 위해 **discounting** 을 도입함.
- 미래 보상에 점점 작은 가중치를 주는 것.

$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

- $\gamma \in [0, 1]$ : **discount rate**. 
- $\gamma < 1$ 이고 보상이 유계라면 무한급수가 수렴함.
    - <span>$$ G_t = \sum_{k=0}^{\infty} \gamma^k \cdot 1 = \frac{1}{1 - \gamma} $$</span>
- $\gamma = 0$ 이면 myopic agent, $\gamma$ 가 1에 가까울수록 멀리 내다봄.

### Recursive 관계

이후 모든 이론의 토대가 되는 재귀식은 다음과 같이 얻어짐.

$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \dots \\
&= R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \dots) \\
&= R_{t+1} + \gamma G_{t+1}
\end{aligned}
$$
- return을 한 번에 다 계산하지 않고 **재귀적으로** 표현할 수 있음

---

## Unified Notation: Absorbing State

### 문제점

| 유형       | Return 정의                                    | $T$      | $\gamma$ |
| ---------- | ---------------------------------------------- | -------- | -------- |
| Episodic   | $G_t = \sum_{k=t+1}^{T} R_k$                   | 유한     | 보통 1   |
| Continuing | $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ | $\infty$ | < 1      |

- episodic task에서는 합이 유한, continuing task에서는 무한합이 됨. 
- 두 경우를 한 식으로 묶으려면 episode 종료 시 자기 자신으로 전이하면서 0 보상을 내는 **absorbing state** 를 도입하면 됨.

### 해결책 : Absorbing State

absorbing state
- Episodic task에서 episode가 끝나면, 거기서 진짜 끝내는 게 아니라 **자기 자신으로만 전이하면서 보상 0을 내뱉는 가상의 상태** 를 추가 


```
S_0 ──(R_1=+1)──► S_1 ──(R_2=+1)──► S_2 ──(R_3=+1)──► [■] ──(0)──► [■] ──(0)──► ...
                                                       ↑              ↑
                                                       │              │
                                                  여기가 원래         이후로는
                                                  종료 시점          무한히 0
```

**왜 이게 잘 작동하는가?**

원래 episode의 보상이 $+1, +1, +1$ 이었다고 하자.  
Absorbing state를 도입하면 그 뒤로 $0, 0, 0, \dots$ 이 무한히 이어짐.  

Return을 계산해보면:

$$ G_0 = 1 + 1 + 1 + 0 + 0 + \dots = 3 $$

- 원래 episode의 보상 총합과 정확히 같음
- Discount를 써도 마찬가지 — 0에 $\gamma^k$ 를 곱해봤자 0이니까.

이렇게 보면 $T$ 개를 더하든 무한개를 더하든 같은 return을 얻음. 따라서 다음 한 식으로 충분함.

$$
G_t \doteq \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k
$$

- 여기서 $T = \infty$ 또는 $\gamma = 1$ 이 될 수 있음.

---

## Policies and Value Functions

### Policy

지금까지 agent가 "행동을 선택"한다고만 말했지, *어떻게* 선택하는지는 정의하지 않았음.  
이걸 정의한 게 **policy**.

**Policy** $\pi$ 는 "상태 → 행동 확률" 의 매핑:

$$ \pi(a \mid s) = \Pr{A_t = a \mid S_t = s} $$

- $\pi(a \mid s)$: 상태 $s$ 에 있을 때 행동 $a$ 를 선택할 확률
- 확률론적
    - 결정론적 policy(상태 보면 무조건 같은 행동)는 그 특수 케이스.
- 예를 들어 가위바위보에서 무조건 가위만 내면 상대에게 읽히니까, 확률적으로 섞는 게 유리한 경우가 있음.

**예시 — 청소 로봇:**

| 상태 | $\pi(\text{search} \mid s)$ | $\pi(\text{wait} \mid s)$ | $\pi(\text{recharge} \mid s)$ |
| ---- | --------------------------- | ------------------------- | ----------------------------- |
| high | 0.7                         | 0.3                       | (선택 불가)                   |
| low  | 0.2                         | 0.3                       | 0.5                           |

이런 표 하나가 policy 하나. 강화학습의 목표는 결국 *좋은 policy* 를 찾는 것.

---
### Value Function

value function : 어떤 상태 $s$에 있는 것이 얼마나 좋은가?
- 미래 보상은 어떤 행동을 하느냐 (= 어떤 policy를 따르느냐) 에 따라 달라짐
- value는 항상 policy에 대해 정의됨

#### State-value function $v_\pi(s)$

state-value function : policy $\pi$ 를 따랐을 때 상태 $s$ 에서 시작해 얻게 될 return의 기댓값

$$
v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\middle|\, S_t = s\right]
$$
- policy $\pi$ 하에서 상태 $s$ 의 가치

#### Action-value function $q_\pi(s, a)$

상태 $s$ 에서 행동 $a$ 를 취한 후 policy $\pi$ 를 따랐을 때의 return 기댓값.

$$
q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\middle|\, S_t = s,\, A_t = a\right]
$$

**$v_\pi$ vs $q_\pi$ 의 차이**:
- $v_\pi(s)$: 현재 state에서 시작해서 앞으로 계속 모든 액션을 policy를 따랐을 때 총 return의 기댓값 
- $q_\pi(s, a)$: 현재 상태 s에서 액션 a를 한 번 지정하고, 그 다음부터는 계속 polcy $\pi$를 따를 때 총 return의 기댓값

왜 둘 다 필요한가?
- 한 번만 다른 행동을 했을 때 return이 어떻게 변하는가 를 알 수 있음
- $q_\pi$ 가 있으면 "어떤 행동을 해야 좋은가?"를 바로 비교할 수 있음.
- $v_\pi$ 만 있으면 그게 안 됨.
- 이 비교가 강화학습에서 policy improvement의 핵심
    - 현재 policy가 잘하는지 못하는지를 행동별로 따져보고, 더 나은 행동이 있으면 그걸로 policy를 갱신
    - 어떤 상태에서든 $\pi$ 보다 한 번만 다르게 행동해도 이득이 없다면 이미 $\pi$는 optimal
    - 어떤 상태에서 한 번 다르게 행동하니 이득이면 그 방향으로 $\pi$를 고치면 항상 더 좋아짐

<br>
**예시**

$s = $ "high (배터리 충분)" 일 때 $\pi$ 가 search 0.7, wait 0.3 라면,
- $q_\pi(\text{high}, \text{search})$: search를 한 후의 가치 — 보통 더 높을 것
- $q_\pi(\text{high}, \text{wait})$: wait을 한 후의 가치 — 보통 더 낮을 것
- $v_\pi(\text{high})$: 둘의 가중평균 = $0.7 \cdot q_\pi(\text{high}, \text{search}) + 0.3 \cdot q_\pi(\text{high}, \text{wait})$

#### 두 value function의 관계

$v_\pi$ 와 $q_\pi$ 는 서로 변환 가능함.

**$v_\pi$ 를 $q_\pi$ 로 표현**:

$$ v_\pi(s) = \sum_a \pi(a \mid s) , q_\pi(s, a) $$

- 상태의 가치 = 가능한 행동들의 가치를 policy로 가중평균

**$q_\pi$ 를 $v_\pi$ 로 표현**:

$$ q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\big[r + \gamma v_\pi(s')\big] $$

- (상태, 행동)의 가치 = 즉시 보상 기댓값 + 할인된 다음 상태 가치 기댓값

---

### Bellman equation 유도

value function $v_\pi$ 는 어떤 *재귀식* 을 만족함. "지금 상태의 가치"와 "다음 상태들의 가치" 사이에 일관성 관계가 있다는 것.

이 재귀식을 **Bellman equation** 이라고 부름.


**시작점**: $v_\pi$ 의 정의

$$ v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] $$

**Step 1: Return 재귀식 적용**

앞에서 유도한 $G_t = R_{t+1} + \gamma G_{t+1}$ 을 대입:

$$ v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s] $$

**Step 2: 기댓값을 풀어쓰기**

이제 기댓값을 명시적인 합으로 풀어야 함.  
"policy $\pi$ 를 따르고 현재 상태가 $s$ 일 때" 일어날 수 있는 모든 일을 가능한 분기별로 나열:

- 먼저, agent는 policy $\pi$ 에 따라 행동 $a$ 를 확률 $\pi(a \mid s)$ 로 선택
- 그 다음, 환경이 dynamics $p$ 에 따라 다음 상태 $s'$ 와 보상 $r$ 을 확률 $p(s', r \mid s, a)$ 로 결정

이 두 단계를 다 합쳐서:

$$ v_\pi(s) = \underbrace{\sum_a \pi(a \mid s)}_{\substack{\text{policy로}\ \text{행동 선택}}} \underbrace{\sum_{s'} \sum_r p(s', r \mid s, a)}_{\substack{\text{환경이}\ s', r \text{ 결정}}} \left[\underbrace{r}_{\substack{\text{즉시}\ \text{보상}}} + \underbrace{\gamma \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']}_{\text{다음 상태에서의 기대 return}}\right] $$


**Step 3: 안쪽 기댓값을 $v_\pi$ 로 치환**

안쪽의 $$\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$$ 는 **$v_\pi(s')$ 의 정의** 와 같음.

$$ \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] = v_\pi(s') $$를 대입하면 

$$ v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_\pi(s')\right]$$


이 식이 바로 $v_\pi$ 에 대한 **Bellman equation**. 

어떤 상태의 가치가, 다음 상태의 가치와 즉시 보상의 가중평균과 같아야 한다는 self-consistency 조건.

#### 예시
```
┌─────────┬─────────┬─────────┐
│         │  +2.3   │         │
│    .    │     ↑   │    .    │
├─────────┼─────────┼─────────┤
│  +0.7  ←│    s    │→  +0.4  │
│         │v(s)=+0.7│         │
├─────────┼─────────┼─────────┤
│         │    ↓    │         │
│    ·    │   -0.4  │    ·    │
└─────────┴─────────┴─────────┘
```
격자 공간에서 가운데 행 가운데 칸의 가치가 +0.7이고, 이웃 네 칸 가치가 위와 같다고 할 때  
Random policy ($\pi(a \mid s) = 0.25$ for each $a$) 하에서 보상은 모두 0, $\gamma = 0.9$ 이면:

$$ v_\pi(s) = \sum_a 0.25 \cdot \big[0 + 0.9 \cdot v_\pi(s')\big] $$

각 방향으로 가면 결정론적으로 해당 이웃으로 가니까:

$$ v_\pi(s) = 0.25 \cdot 0.9 \cdot (2.3 + 0.4 + (-0.4) + 0.7) = 0.25 \cdot 0.9 \cdot 3.0 = 0.675 \approx 0.7 $$
- 진짜로 자신의 가치 +0.7과 일치


#### Bellman equation의 의의
1. **재귀 구조**: 미래 무한대까지 안 봐도, 한 스텝 만 보면 가치를 계산할 수 있음. 이게 모든 RL 알고리즘의 기초.
2. **유일해**: 이 방정식은 (finite MDP에서) $v_\pi$ 가 유일한 해. 즉 풀면 답이 나옴.
3. **알고리즘 도출**: 이 식을 어떻게 근사적으로 푸느냐에 따라 dynamic programming, Monte Carlo, TD learning 등 다양한 알고리즘이 갈라져 나옴.

---

### Backup diagram

Bellman equation은 backup diagram으로 시각화할 수 있음.
```
                    s              ← root: 현재 상태
                   /|\
                  / | \             ← policy π 가 행동 a 선택 π(a|s)
                 /  |  \
                •   •   •           ← 행동 노드 (검은 점) = (s, a) 쌍
               /|\ /|\ /|\
              / | × | × | \         ← 환경 p 가 다음 상태 s' 결정 (환경 dynamics에 따라 분기) 
             ○  ○  ○  ○  ○  ○     ← 다음 상태 노드 (흰 원)
```
- 각 빈 원은 상태, 검게 칠해진 원은 (state, action) 쌍을 나타냄
- 위쪽 분기 (상태 → 행동): policy $\pi(a \mid s)$ 가 결정
- 아래쪽 분기 (행동 → 다음 상태): 환경 dynamics $p(s', r \mid s, a)$ 가 결정
- 가치가 후속 상태로부터 "백업"되어 현재 상태로 흘러간다는 뜻에서 backup diagram이라 부름.

---

### Action-value의 Bellman equation

$q_\pi$ 도 비슷하게 유도됨. 출발점은 다른데:

$$ q_\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] $$

여기서 $(s, a)$ 가 이미 고정되어 있으니까, policy의 합 $\sum_a$ 가 빠지고 환경 분기만 펼침:

$$ q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']\Big] $$

이제 안쪽을 $v_\pi(s')$ 로 적든, 아니면 다시 한 번 policy로 펼쳐서 $q_\pi(s', a')$ 로 적든 가능함. 후자로 쓰면:

$$ q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma \sum_{a'} \pi(a' \mid s'), q_\pi(s', a')\Big] $$


```
              (s, a)               ← 시작: 상태-행동 쌍
              / | \
             /  |  \               ← 환경이 r, s' 결정
            ○   ○   ○              ← 다음 상태 노드
           /|\ /|\ /|\
          •  • •  • • •         ← policy π 가 다음 행동 a' 선택
```

$v_\pi$ 의 backup diagram과 비교하면 
- 시작점이 행동 노드라는 점
- 분기 순서가 "환경 먼저, policy 나중"으로 뒤집힌 점

이 다름






---

## Optimal Policies and Optimal Value Functions

지금까지는 어떤 policy $\pi$ 가 주어졌을 때 그 가치 $v_\pi$ 를 평가하는 법을 봤음.  
그런데 강화학습의 진짜 목표는 *가장 좋은* policy를 찾는 것.  
그러려면 먼저 "$\pi_1$ 이 $\pi_2$ 보다 좋다"는 게 뭔지 정의해야 함.

### Policy 간의 순서

**정의**: 모든 상태에서 $\pi$ 의 가치가 $\pi'$ 의 가치보다 크거나 같으면 $\pi \geq \pi'$.

$$ \pi \geq \pi' \iff v_\pi(s) \geq v_{\pi'}(s) \quad \text{for all } s \in \mathcal{S} $$

- Finite MDP에는 항상 **모든 policy보다 좋거나 같은** policy가 적어도 하나 존재함. 이걸 **optimal policy** 라 부르고 $\pi_*$ 로 표기함.
- optimal policy는 여러 개일 수 있지만 모두 같은 state-value function을 공유함.

---

### Optimal value functions

$$
v_*(s) \doteq \max_\pi v_\pi(s)
$$
- 상태 $s$ 에서 *최선의* policy를 따랐을 때 얻을 수 있는 return의 기댓값
- $*$ 는 "optimal" 의 표준 기호

$$
q_*(s, a) \doteq \max_\pi q_\pi(s, a)
$$
- 상태 $s$ 에서 행동 $a$ 를 취한 뒤 *최선의* policy를 따랐을 때의 return 기댓값

$q_\*$ 는 $v_\*$ 로 표현됨.

$$
q_*(s, a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a]
$$
- 행동 $a$ 를 하면 즉시 보상 $R_{t+1}$ 을 받고 다음 상태 $S_{t+1}$ 로 감
- 그 다음부터는 optimal policy를 따르니까 $v_*(S_{t+1})$ 만큼의 가치를 가짐
- 둘을 합쳐서 기댓값

---

### Bellman optimality equation


일반 Bellman equation은:
$$ v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\big[r + \gamma v_\pi(s')\big] $$

- $\sum_a \pi(a \mid s)$ 는 policy에 따라 행동들을 가중평균 하는 부분
- 직관적으로 optimal하려면 가장 좋은 행동을 골라야 함
- 가중평균이 아니라 max를 취함

따라서 $v_*$ 의 Bellman equation에서는 $\sum_a \pi(a \mid s)$ 가 $\max_a$ 로 바뀜.

#### 유도

**Step 1**: $v_\*$ 와 $q_\*$ 의 관계에서 출발.

Optimal policy 하에서는 무조건 가장 좋은 행동을 고르니까:

$$ v_*(s) = \max_{a \in \mathcal{A}(s)} q_{\pi_*}(s, a) = \max_a q_*(s, a) $$

- 상태 $s$ 의 optimal 가치 = 그 상태에서 취할 수 있는 행동들의 optimal 가치 중 최댓값

**Step 2**: $q_\*$ 의 정의 풀어쓰기.

$q_*(s, a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]$ 였고, 일반 Bellman 유도 때처럼 $G_t = R_{t+1} + \gamma G_{t+1}$ 대입:

$$ v_*(s) = \max_a \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] $$

**Step 3**: 안쪽 $G_{t+1}$ 의 기댓값을 $v_\*$ 로 치환.

행동 $a$ 를 한 뒤로는 optimal policy를 따른다 는 조건이므로, $\mathbb{E}[G_{t+1} \mid S_{t+1} = s'] = v_*(s')$:

$$ v_*(s) = \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a] $$

**Step 4**: 기댓값을 환경 dynamics로 풀어쓰기.

$(s, a)$ 가 주어지면 $(s', r)$ 은 $p(s', r \mid s, a)$ 에 따라 결정되므로:

$$ v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\big[r + \gamma v_*(s')\big] $$
- 이게 Bellman optimality equation for $v_\*$
- $v_\*$ 도 value function이므로 self-consistency 조건을 만족해야 함.
- 다만 optimal policy를 따르면 *해당 상태에서 가장 좋은 행동* 으로부터 오는 return과 같아야 함.


한 번에 보면
$$
\begin{aligned}
v_*(s) &= \max_{a \in \mathcal{A}(s)} q_{\pi_*}(s, a) \\
&= \max_a \mathbb{E}_{\pi_*}[G_t \mid S_t = s, A_t = a] \\
&= \max_a \mathbb{E}_{\pi_*}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] \\
&= \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a] \\
&= \max_a \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_*(s')\right]
\end{aligned}
$$

- 마지막 두 줄이 $v_*$ 에 대한 **Bellman optimality equation**. 
- 일반 Bellman equation과의 차이는 $\sum_a \pi(a \mid s)$ 가 $\max_a$ 로 바뀌었다는 점뿐.

---

#### Bellman optimality equation for $q_\*$
$q_\*$ 에 대해서도 같은 논리로 다음을 얻음.

$$
\begin{aligned}
q_*(s, a) &= \mathbb{E}\!\left[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \,\middle|\, S_t = s, A_t = a\right] \\
&= \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \max_{a'} q_*(s', a')\right]
\end{aligned}
$$
- **차이점**: max가 식의 안쪽으로 들어감.
- 행동 $a$ 는 이미 주어졌으니 그 다음 행동 $a'$ 에 대해 max를 취함.

---

### Optimality equation의 backup diagram

```
v*:                     q*:
       s                      s, a
      /|\                       |
     max                      r,s'
    / | \                      /  \
   •  •  •                    ○    ○
  /|\ ...                   max   max
 ○ ○  ... ○                 / \   / \
                           •   • •   •
```

- 선택 지점에 호(arc) 표시가 붙어 있다는 점만 일반 backup diagram과 다름. 
- 왼쪽이 $v_\*$, 오른쪽이 $q_\*$ 의 식에 대응함.

---

### Optimal policy 도출

finite MDP에 대해 Bellman optimality equation은 policy와 무관한 유일해를 가짐.  
상태가 $n$ 개면 $n$ 개의 미지수에 대한 $n$ 개의 비선형 방정식.

$v_\*$ 를 알면 optimal policy를 *one-step search* 만으로 만들 수 있음.
- 각 상태에서 Bellman optimality equation의 max를 달성하는 행동에만 0이 아닌 확률을 부여하면 됨.
- $$ \pi_*(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a)\big[r + \gamma v_*(s')\big] $$
- 각 상태에서 one-step look-ahead 하고, 즉시 보상 + $\gamma \times$ 다음 상태의 optimal 가치 의 기댓값이 가장 큰 행동을 고름
- $v_\*$ 자체가 이미 모든 미래 결과를 담고 있음
    - $v_*(s')$ : "$s'$ 에서 시작해 optimal하게 행동하면 받을 모든 미래 보상의 기댓값
    - 한 스텝 미리 본 $v_*(s')$ 값에는 이미 그 뒤 백만 스텝 후의 보상까지 다 반영되어 있음
- 한 스텝만 greedy하게 봐도 장기적으로 optimal함

<br>
$q_\*$ 가 있으면 더 쉬움.  
one-step search조차 필요 없이 $\arg\max_a q_*(s, a)$ 만 보면 됨. 
- 환경 dynamics $p$ 를 몰라도 optimal action을 고를 수 있다는 게 action-value의 큰 장점
- 이래서 Q-learning 같은 model-free 알고리즘이 가능

---

### Bellman Optimality Equation의 의의

1. Policy 없이 표현됨
- 일반 Bellman은 $\pi$ 가 들어가지만, optimal 버전은 $\pi$ 가 사라짐
- policy를 직접 다루지 않고 가치 함수만 다루면 됨

2. 유일해를 가짐
- Finite MDP에 대해 Bellman optimality equation은 **policy와 무관하게 유일한 해** $v_\*$ 를 가짐.
- 상태가 $n$ 개라면 $n$ 개의 비선형 방정식 ($\max$ 때문에 비선형)이 됨.
- 이론적으로는 환경 dynamics $p$ 를 알면 풀 수 있고, 그러면 $v_*$ 가 나옴.


---

## Optimality and Approximation

"$v_\*$ 가 유일해이고 풀 수 있다면, 그냥 풀면 되지 않나?" 싶지만 현실에서는 거의 불가능함.

Bellman optimality equation을 해석적으로 푸는 것은 다음 세 가정을 함.

1. 환경의 dynamics를 정확히 안다 - 보통 모름
2. 충분한 계산 자원이 있다 - state 개수가 많아지면 explode
3. Markov property가 성립한다 - 상태 표현이 부족하면 깨짐

현실에서는 셋 모두 깨지기 일쑤.
- 가령 backgammon은 (1), (3)을 만족하지만 약 $10^{20}$ 개의 상태 때문에 (2)가 깨짐. 
- 강화학습은 결국 *근사해* 를 찾는 작업.

근사가 가능한 한 가지 이유는 agent가 마주치는 상태 분포가 균일하지 않다는 점. 
- 자주 방문하는 상태에 대해서는 좋은 결정을 내리도록 학습 자원을 집중하고, 거의 만나지 않는 상태에서는 약간 suboptimal해도 큰 손해가 아님. 
- 이 online 특성이 강화학습을 다른 MDP 근사 기법과 구분 짓는 핵심.

---

## 정리





```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Agent  ←─ S_t, R_t ─────  Environment                 │
│         ──── A_t ──────►                                │
│                                                         │
│   환경의 모든 것: dynamics p(s', r | s, a)                 │
│   Markov property: 과거는 현재 상태에 다 들어 있음             │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
            Return: G_t = R_{t+1} + γG_{t+1}
                          │
                          ▼
        ┌─────────────────┴────────────────────┐
        ▼                                      ▼
    Policy π                              Optimal π_*
        │                                      │
        ▼                                      ▼
  v_π(s) = E_π[G_t | s]              v_*(s) = max_π v_π(s)
  q_π(s,a) = E_π[G_t | s, a]         q_*(s,a) = max_π q_π(s,a)
        │                                      │
        ▼                                      ▼
   Bellman equation                  Bellman optimality eq.
   (Σ_a π(a|s) 가중평균)              (max_a)
        │                                      │
        └─────────┬───────────────────┬────────┘
                  ▼                   ▼
       정책 평가 (policy evaluation)  최적해는 유일하게 존재
       각종 RL 알고리즘이             하지만 현실에서는 근사해야 함
       이 방정식을 어떻게 푸느냐의 차이
```

| 개념               | 정의                                                         | 한 줄 설명               |
| ------------------ | ------------------------------------------------------------ | ------------------------ |
| Dynamics           | $p(s', r \mid s, a)$                                         | 환경이 어떻게 반응하는지 |
| Return             | $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$               | 미래 누적 보상           |
| Policy             | $\pi(a \mid s)$                                              | 행동 선택 규칙           |
| State value        | $v_\pi(s) = \mathbb{E}_\pi[G_t \mid s]$                      | 그 상태의 가치           |
| Action value       | $q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid s, a]$                | 그 (상태, 행동)의 가치   |
| Bellman eq.        | $v_\pi(s) = \sum_a \pi(a\|s) \sum_{s',r} p(s',r\|s,a)[r + \gamma v_\pi(s')]$ | 가치의 재귀 구조         |
| Bellman optimality | $v_\*(s) = \max_a \sum_{s',r} p(s',r\|s,a)[r + \gamma v_\*(s')]$ | optimal 가치의 재귀 구조 |


