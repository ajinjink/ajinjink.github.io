---
title: "[DB] 인덱스 튜닝 - 인덱스 스캔 효율화"
date: 2025-04-02
categories: ["Database"]
tags: ["index", "index-tuning", "index search", "access predicate", "filter predicate", "index skip scan", "like/between", "nvl/decode"]
use_math: true
---

## 인덱스 탐색

![Image](/assets/images/2025-04-02/Picture1.png)

- 수직적 탐색을 할 때, 찾고자 하는 시작점보다 값이 같거나 큰 키를 만나면, 해당 키 바로 이전 키를 타고 내려가야 하는 이유
- (B, 3)를 찾고 싶으면 루트/브랜치 블록에서 (B, 3)이전 키인 (A, 3)부터 탐색을 시작해서 (B, 3)의 시작점을 찾을 수 있음

```sql
WHERE C1 = 'B'
  AND C2 BETWEEN 2 AND 3
```

- 시작점은 (B, 2), 끝점은 (B, 4)
- C1과 C2가 모두 수평적 탐색의 시작과 끝을 결정짓는 데에 기여

```sql
WHERE C1 BETWEEN 'A' AND 'C'
  AND C2 BETWEEN 2 AND 3
```

- 시작점은 (A, 2), 끝점은 (C, 4)
- C1은 시작과 끝을 결정
- C2는 처음 2부터 시작하는 것과 3에서 끝나는 것 외에, 탐색 중간에 C1 = 'B' 인 구간에서 탐색 범위를 줄이는 데 기여하지 못하고 있음

------

인덱스 스캔을 모두 등치 조건으로 사용하면, 리프 블록을 스캔하면서 읽은 레코드는 모두 테이블 액세스로 이어지므로, 인덱스 스캔 단계에서의 비효율은 없음.
인덱스 칼럼 중 일부가 조건절에 없거나 등치 조건이 아니어도, 뒤쪽 칼럼이면 비효율 없음.
인덱스 선행 칼럼이 조건절에 없거나 부들호, BETWEEN, LIKE 같은 범위검색 조건이면 비효율이 생김.

------

![Image](/assets/images/2025-04-02/Picture2.png)

- C1, C2, C3가 '성능검'인 레코드를 찾을 때는 '성능검사'에서 시작해서 '성능계수'에서 끝남
- C1, C2, C4가 '성', '능', '선'인 레코드를 찾을 때는 모든 레코드 다 읽어야 함
  - 비효율적
  - 인덱스 선행 칼럼(C3)에 조건절이 없기 때문

## 액세스 조건, 필터 조건

**인덱스 액세스 조건**

- 인덱스 스캔 범위 결정 (시작과 끝)

**인덱스 필터 조건**

- 테이블로 액세스할지 말지 결정

위 인덱스에서 '성능검'을 찾을 때는 C1, C2, C3가 모두 인덱스 액세스 조건. 스캔 범위 결정.
'성능', '선'을 찾을 때는 C1, C2는 인덱스 액세스 조건, C3는 인덱스 필터 조건.

**테이블 필터 조건**

- 쿼리 수행 다음 단계로 전달하거나 최종 결과집합에 포함할지 결정

![Image](/assets/images/2025-04-02/Picture3.png)

cf) 옵티마이저에서 계산하는 인덱스를 이용한 테이블 액세스 비용
= 인덱스 수직적 탐색 비용 + 인덱스 수평적 탐색 비용 + 테이블 랜덤 액세스 비용
= 인덱스 루트/브랜치 에서 읽는 블록 수 + 인덱스 리프 블록을 스캔할 때 읽는 블록 수 + 테이블 액세스 과정에서 읽는 블록 수

------

## 비교 연산자와 칼럼 순서에 따른 군집성

인덱스는 정렬이 되어 있음. 같은 값을 갖는 레코드들이 서로 군집해 있음.
인덱스 칼럼을 앞쪽부터 누락없이 '=' 연산자로 조회하면 조건절을 만족하는 레코드는 모여 있음.
![Image](/assets/images/2025-04-02/Picture4.png)

```sql
where C1 = 1
and C2 = 'A'
and C3 = '나'
and C4 = 'a'
```

```sql
where C1 = 1
and C2 = 'A'
and C3 = '나'
and C4 >= 'a'
```

- 비교 조건이 모두 '='일 때와 마지막 칼럼만 범위검색 조건일 때는 조건을 만족하는 레코드가 서로 모여 있음

```sql
where C1 = 1
and C2 = 'A'
and C3 between '가' and '다'
and C4 = 'a'
```

- C1, C2 조건을 만족하는 인덱스 레코드는 모여 있음
- C3, C4 조건까지 만족하는 레코드는 흩어져 있음

- 선행 칼럼이 모두 '=' 조건인 상태에서, 첫 번째 나타나는 범위검색 조건까지만 만족하는 인덱스 레코드는 모여있고,
- 그 이하 조건가지 만족하는 레코드는 비교 연산자 종류에 상관없이 흩어짐

- 첫 번째 나타나는 범위검색 조건이 인덱스 스캔 범위 결정
  - 여기까지가 인덱스 액세스 조건
- 그 이후 칼럼은 인덱스 범위를 결정짓지 못하고 흩뿌려져 있음
  - 인덱스 필터 조건

------

## BETWEEN을 IN-List로 전환

![Image](/assets/images/2025-04-02/Picture5.png)
[인터넷매물, 아파트시세코드, 평형, 평형타입] 으로 인덱스가 구성되어 있을 때

```sql
select 층, 평당가, 입력일, 동, 매물구분, 연사용일수, 중개업소코드
from 매물아파트매매
where 인터넷매물 between '1' and '3'
and 아파트시세코드='900056'
and 평형 = '59'
and 평형타입 = 'A'
order by 입력일 desc
```

- 선행 칼럼이 범위 검색 → 비효율

```sql
select 층, 평당가, 입력일, 동, 매물구분, 연사용일수, 중개업소코드
from 매물아파트매매
where 인터넷매물 in ('1', '2', '3')
and 아파트시세코드='900056'
and 평형 = '59'
and 평형타입 = 'A'
order by 입력일 desc
```

- between 조건을 IN-List로 바꾸면 수직적 탐색 3번
- 모든 조건절을 '='로 사용하고, 3개 union all한 것과 같음

IN-List 항목 개수가 늘어난다면 BETWEEN을 IN-List로 바꾸기 힘들 수도.
NL 조인이나 서브쿼리로 구현하면 됨.

```sql
select /*+ ordered use_nl(b) */ 
				b.층, b.평당가, b.입력일, b.동, b.매물구분, b.연사용일수, b.중개업소코드
from 통합코드 a, 매물아파트매매 b
where a.코드구분 = 'CD064'
and a.코드 between '1' and '3'
and b.인터넷매물 = a.코드
and b.아파트시세코드='900056'
and b.평형 = '59'
and b.평형타입 = 'A'
order by b.입력일 desc
```

- IN-List 값들을 코드 테이블로 관리하고 있을 때 위와 같이 가능
- '인터넷매물'을 '=' 조건으로 검색 중

------

BETWEEN을 IN-List로 바꿀 때, IN-List 개수가 너무 많으면 안 됨.
![Image](/assets/images/2025-04-02/Picture6.png)

- 수직적 탐색이 많이 발생
- BETWEEN 조건 때문에 리프 블록을 많이 스캔하는 비효율 < IN-List 개수만큼 브랜치 블록을 반복 탐색하는 비효율
- depth가 깊으면 반복적으로 수직적 탐색 하는 게 더 비효율적

IN-List 전환은 인덱스 스캔 과정에 선택되는 레코드들이 서로 멀리 떨어져 있을 때만 유용

```sql
where 고객등급 between 'C' and 'D'
and 고객번호 = 123
```

- [고객등급, 고객번호] 순으로 인덱스가 구성되어 있을 때
- 인덱스 선두 칼럼이 BETWEEN 조건절이기 때문에, 고객번호 = 123을 만족하는 레코드는 멀리 떨어져 있음

만약 조건을 만족하는 레코드 2건이 서로 가까이 붙어있다면, 둘 사이에 놓인 인덱스 블록이 매우 소량

- IN-List로 변환해도 효과가 없거나, 수직적 탐색 때문에 오히려 블록 I/O가 더 많이 발생

------

## Index Skip Scan 활용

월별고객판매집계 테이블에 2018년 1월-12월 판매데이터가 월별 10만건씩 있다고 했을 때

```sql
select count(*)
from 월별고객별판매집계
where 판매구분 = 'A'
and 판매월 between '201801' and '201812'
```

- 위 쿼리를 최적으로 수행하려면 [판매구분, 판매월] 순으로 인덱스를 구성해야 함

[판매구분, 판매월] 순으로 IDX1을 구성했을 때

```
Rows   Row Source Operation 
------ ----------------------------------------------------------------
     1 SORT AGGREGATE (cr=281 pr=0 pw=0 time=47753 us)
100000  INDEX RANGE SCAN 월별고객별판매집계_IDX1 (cr=281 pr=0 pw=0 time=...)
```

- IDX1 스캔하면서 281개의 블록 I/O 발생
- 테이블 액세스는 발생하지 않음 (table access / full table scan 같은 거 없어)

![Image](/assets/images/2025-04-02/Picture7.png)

판매구분 'A'인 레코드가 각 월 앞쪽에 소량 위치하고 있을 때
[판매월, 판매구분] 순으로 구성된 인덱스 IDX2를 사용하면

```
Rows   Row Source Operation 
------ ----------------------------------------------------------------
     1 SORT AGGREGATE (cr=3090 pr=0 pw=0 time=206430 us)
100000  INDEX RANGE SCAN 월별고객별판매집계_IDX2 (cr=3090 pr=0 pw=0 time=...)
```

- 테이블 액세스가 없음에도 불구하고 I/O 많이 발생
- 인덱스 선두 칼럼이 BETWEEN 조건이라서, 판매구분 = 'B'인 레코드까지 모두 스캔했기 때문

BETWEEN을 IN-List로 바꿔서 다음과 같이 실행하면

```sql
select /*+ index(t 월별고객별판매집계_IDX2) */ count(*)
from 월별고객별판매집계 t
where 판매구분 = 'A'
and 판매월 in ('201801', '201802', '201803', '201804', '201805', '201806',
             '201807', '201808', '201808', '201810', '201811', '201812')
```

```txt
Rows   Row Source Operation 
------ -----------------------------------------------------------------
     1 SORT AGGREGATE (cr=314 pr=0 pw=0 time=31527 us)
100000  INLIST ITERATOR (cr=314 pr=0 pw=0 time=900030 us)
100000   INDEX RANGE SCAN 월별고객별판매집계_IDX2 (cr=3090 pr=0 pw=0 time=...)
```

- 블록 I/O 개수 314개로 감소
- 인덱스 브랜치 블록을 12번 반복 탐색했지만, 리프 블록을 스캔할 때의 비효율을 제거
- 성능 10배 향상

Index Skip Scan 사용하면

```sql
select /*+ INDEX_SS(t 월별고객별판매집계_IDX2) */ count(*)
from 월별고객별판매집계 t
where 판매구분 = 'A'
and 판매월 between '201801' and '201812'
```

```
Rows   Row Source Operation 
------ -----------------------------------------------------------------
     1 SORT AGGREGATE (cr=300 pr=0 pw=0 time=94282 us)
100000  INDEX SKIP SCAN 월별고객별판매집계_IDX2 (cr=300 pr=0 pw=0 time=500073 us)
```

- 인덱스 선두 칼럼이 BETWEEN 조건인데도 300블록만 읽고 끝
- [판매구분, 판매월]로 구성된 인덱스 사용할 때와 큰 차이 없음

------

## IN 조건은 ‘=’인가

고객별가입상품 테이블에서 고객번호의 평균 카디널리티가 3이라고 가정 (고객별로 평균 3건의 상품을 가입)

```sql
select *
from 고객별가입상품
where 고객번호 = :cust_no
and 상품ID in ('NH00037', 'NH00041', 'NH00050')
```

인덱스를 [상품ID, 고객번호] 순으로 생성하면
같은 상품은 고객번호 순으로 정렬된 상태로 리프 블록에 저장됨.
고객번호 입장에서는 상품ID에 따라 흩어져 있음.

![Image](/assets/images/2025-04-02/Picture8.png)

- 고객번호 = 1234를 만족하는 레코드는 서로 멀리 떨어져 있음
- IN-List Iterator 방식으로 액세스하는 게 효휼적 (IN 조건을 개별 조건으로 분리해서 각각 처리)

```sql
select *
from 고객별가입상품
where 고객번호 = :cust_no
and 상품ID = 'NH00037'
union all
select *
from 고객별가입상품
where 고객번호 = :cust_no
and 상품ID = 'NH00041'
union all
select *
from 고객별가입상품
where 고객번호 = :cust_no
and 상품ID = 'NH00050'
```

- 이때는 IN 조건이 '='
- 고객번호와 상품ID가 둘 다 인덱스 액세스 조건으로 사용됨
- 인덱스를 수직적으로 3번 탐색 (9개 블록 읽음)

상품ID in ('NH00037', 'NH00041', 'NH00050') 를 사용하는 원래 쿼리에서는

- 인덱스를 정상적으로 탈려면 스캔 시작점을 찾아야 하는데, 상품ID에 대해서 시작점 한 지점을 찾을 수 없음
- 상품ID가 선두 칼럼인 상황에서 IN-List Iterator 로 풀지 않으면, 상품ID는 필터 조건
  - 인덱스나 테이블 전체를 스캔하면서 필터링 해야 함

------

인덱스를 [고객번호, 상품ID] 순으로 생성하면
같은 고객은 상품ID 순으로 정렬된 상태로 같은 리프 블록에 저장됨.
![Image](/assets/images/2025-04-02/Picture9.png)

- 이걸 IN-List Iterator로 풀면 인덱스 3번 수직적으로 탐색하면서 블록 9개 읽어야 함.
- 이걸 IN-List Iterator로 풀지 않으면 상품ID 조건절은 필터로 처리함.
  - 고객번호만 액세스 조건이므로, 고객번호 = 1234인 레코드를 모두 스캔.
  - 같은 고객은 연속적으로 모여 있으므로, 블록 I/O는 수직적 탐색을 포함해서 블록을 총 3개(or 4)만 읽음
- 이때는 IN 조건이 '='가 아님

IN 조건이 '='이 되러면 IN-List Iterator 방식으로 풀려야 함.
아니면 IN 조건은 필터 조건.

- 위 시나리오에서 상품ID가 액세스 조건으로 의미있는 역할을 하려면 (스캔 범위를 줄이는 데 기여하려면)
  고객별 상품 데이터가 아주 많아야 함.
- 그렇지 않은 상황에서는 상품ID는 필터로 처리되는 게 나음

------

### 인덱스 칼럼 일부만 사용

[고객번호, 상품ID] 순으로 인덱스를 구성한 상황에서
고객번호만 인덱스 액세스 조건으로 사용하려면

```sql
select /*+ num_index_keys(a 고객변가입상품_X1 1) */ *
from 고객별가입상품 a
where 고객번호 = :cust_no
and 상품ID in ('NH00037', 'NH00041', 'NH00050')
```

- num_index*_*keys (테이블, 인덱스, 몇 번째 칼럼까지 액세스 조건으로 사용)
- 고객번호 까지는 액세스 조건으로 사용하고, 상품ID는 필터 조건으로 사용해라

힌트를 사용하지 않고 인덱스 칼럼을 가공할 수도 있음

```sql
select *
from 고객별가입상품
where 고객번호 = :cust_no
and RTRIM(상품ID) in ('NH00037', 'NH00041', 'NH00050')

select *
from 고객별가입상품
where 고객번호 = :cust_no
and 상품ID || '' in ('NH00037', 'NH00041', 'NH00050')
```

- 인덱스 칼럼을 가공해서 일부러 해당 칼럼을 인덱스 스캔에서 제외

상품ID까지 인덱스 액세스 조건으로 사용하고 싶으면 힌트 매개변수 조정

```sql
select /** num_index_keys(a 고객별가입상품_X1 2) */ *
from   고객별가입상품 a
where  고객번호 = :cust_no
and    상품ID in ('NH00037', 'NH00041', 'NH00050')
```

```
Execution Plan
-----------------------------------------------------------------
 0      SELECT STATEMENT Optimizer=ALL_ROWS
 1  0    INLIST ITERATOR
 2  1      TABLE ACCESS (BY INDEX ROWID BATCHED) OF '고객별가입상품' (TABLE)
 3  2        INDEX (RANGE SCAN) OF '고객별가입상품_X1' (INDEX)

Predicate information (identified by operation id):
-----------------------------------------------------------------
   3 - access("고객번호"= TO_NUMBER(:CUST_NO)) AND ("상품ID"='NH00037' OR "상품ID"='NH00041' OR "상품ID"='NH00050'))
```

- 상품ID가 IN-List Iterator 방식으로 풀림

------

## BETWEEN과 LIKE 스캔 범위 비교

BETWEEN과 LIKE 모두 인덱스 스캔에서 비효율 원리가 똑같이 적용되지만, 데이터 분포와 조건절 값에 따라 인덱스 스캔량이 다를 수 있음.
[판매월, 판매구분] 순으로 인덱스가 구성되어 있을 때 다음 두 조건절을 비교하자.

```sql
where 판매월 between '201901' and '201912'
and 판매구분 = 'B'

where 판매월 like '2019%'
and 판매구분 = 'B'
```

판매구분 칼럼에 값이 'A' 90%, 'B' 10% 존재할 때 스캔 범위가 다음과 같다.

![Image](/assets/images/2025-04-02/Picture10.png)

- between은 판매월 = '201901', 판매구분 = 'B'인 첫 번째 레코드에서 스캔 시작
- like는 '201901'을 만나면 그 직전으로 가서 '201900'을 찾아야 하기 때문에 판매구분 = 'B'인 지점부터 시작할 수 없음

판매구분 칼럼에 값이 'A' 10%, 'B' 90% 존재하고, 다음과 같이 조건을 잡으면

```sql
where 판매월 between '201901' and '201912'
and 판매구분 = 'A'

where 판매월 like '2019%'
and 판매구분 = 'A'
```

![Image](/assets/images/2025-04-02/Picture11.png)

- between은 판매월 = '201912', 판매구분 = 'B'인 레코드를 만나면 스캔 종료
- like는 판매월 '201913'의 판매구분 = 'A'도 탐색해야 하기 때문에 중간에 멈출 수 없음.

------

## 범위검색 남용

회사코드, 지역코드, 상품명 등을 입력하는 ‘가입상품’ 테이블이 있을 때
조회할 때 회사코드는 반드시 입력하지만 지역코드는 입력하지 않을 수도 있음.
두 경우를 나눠서 처리하면 쿼리 2개에 대해서 인덱스 스캔 범위가 다름.

```sql
select 고객ID, 상품명, 지역코드, ...   -- 지역코드 입력한 경우
from 가입상품
where 회사코드 = :com
and 지역코드 = :reg
and 상품명 like :prod || '%'

select 고객ID, 상품명, 지역코드, ...   -- 지역코드 입력하지 않은 경우
from 가입상품
where 회사코드 = :com
and 상품명 like :prod || '%'
```

![Image](/assets/images/2025-04-02/Picture12.png)

- 지역코드 조건이 있을 때는 세 칼럼 모두 액세스 조건

이 두 쿼리를 하나의 쿼리로 합치면 다음과 같음

```sql
select 고객ID, 상품명, 지역코드, ...
from 가입상품
where 회사코드 = :com
and 지역코드 like :reg || '%'
and 상품명 like :prod || '%'
```

![Image](/assets/images/2025-04-02/Picture13.png)

- 지역코드를 입력한 경우에 대한 인덱스 스캔 범위가 늘어났음
- 상품명이 액세스 조건이었는데, 필터 조건으로 바뀌었기 때문

이와 비슷하게 필수 조건과 옵션인 조건들을 모두 하나의 쿼리로 표현하기 위해 between을 남용하는 경우도 있음.

- 종목코드 6자리인데 입력 안 됐으면 between '______' and ‘ZZZZZZ’로 표시한다든지

------

## 다양한 옵션 조건 처리

### OR 조건 활용

```sql
select *
from 거래
where (:cust_id is null or 고객ID = :cust_id)
and 거래일자 between :dt1 and :dt2
```

```
Execution Plan
------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS
1    0   TABLE ACCESS (FULL) OF '거래' (TABLE)
```

- 옵션 조건 칼럼을 선두에 두고 [고객ID, 거래일자] 순으로 인덱스를 구성하면, 인덱스 사용 불가
- 인덱스 선두 칼럼에 대한 옵션 조건에 OR 조건 사용하면 안 됨
- [거래일자, 고객ID] 순으로 구성한 인덱스는 사용 가능
  - 고객ID는 필터 조건으로 사용

```
Execution Plan
-----------------------------------------------------------------
 0      SELECT STATEMENT Optimizer=ALL_ROWS
 1  0    FILTER
 2  1      TABLE ACCESS (BY INDEX ROWID) OF '거래' (TABLE)
 3  2        INDEX (RANGE SCAN) OF '거래_IDX3' (INDEX)
-----------------------------------------------------------------

Predicate information (identified by operation id):
-----------------------------------------------------------------
 1 - filter(TO_DATE(:DT1)<=TO_DATE(:DT2))
 2 - filter(:CUST_ID IS NULL OR "고객ID"=TO_NUMBER(:CUST_ID))
 3 - access("거래일자">=:DT1 AND "거래일자"<=:DT2)
```

1.거래일자 조건으로 인덱스를 통해 행을 찾고

2.해당 행의 ROWID로 테이블에 액세스한 후에

3.테이블 데이터에서 고객ID 조건을 필터링

- 인덱스 스캔 단계에서 필터링해도 비효율적인데, 테이블 액세스 단계에서 필터링 중
- 거래일자 between 조건 찾기 위해 인덱스에서 100만 건 스캔하면, 그만큼 테이블 랜덤 액세스 후에 고객ID를 필터링
- 인덱스에 포함되지 않은 칼럼에 대한 옵션 조건은 어차피 테이블에서 필터링할 수밖에 없으므로 OR 사용해도 무방



OR 조건을 활용한 옵션 조건 처리

- 인덱스 액세스 조건으로 사용 불가
  - OR 조건은 명확한 시작점을 결정할 수 없어 인덱스 액세스 조건으로 활용할 수 없음

- 인덱스 필터 조건으로도 사용 불가
  - OR 조건의 한쪽이 true면 다른 쪽은 평가할 필요가 없으므로, 데이터베이스가 인덱스 스캔 도중에 효율적으로 필터링하기 어려움

- 테이블 필터 조건으로만 사용 가능
- 인덱스 칼럼 중 하나 이상이 not null 칼럼이면, 인덱스 필터 조건으로 사용 가능 (from version 18c)



OR 조건의 장점은 옵션 조인 칼럼이 null 허용 칼럼이더라도 결과집합을 보장한다는 것.

```sql
select *
from 거래
where 고객ID = :cust_id
and ((:dt_type = 'A' and 거래일자 between :dt1 and :dt2)
     or
     (:dt_type = 'B' and 결제일자 between :dt1 and :dt2))
```

```
Execution Plan
-----------------------------------------------------------------
 0      SELECT STATEMENT Optimizer=ALL_ROWS
 1  0     CONCATENATION
 2  1       FILTER
 3  2         TABLE ACCESS (BY INDEX ROWID) OF '거래' (TABLE)
 4  3           INDEX (RANGE SCAN) OF '거래_IDX1' (INDEX) -- 고객ID + 거래일자
 5  1       FILTER
 6  5         TABLE ACCESS (BY INDEX ROWID) OF '거래' (TABLE)
 7  6           INDEX (RANGE SCAN) OF '거래_IDX12' (INDEX) -- 고객ID + 결제일자
-----------------------------------------------------------------
```

- 위와 같은 OR 조건절 형태에는 OR-Expansion을 통해 인덱스 사용이 가능

------

### LIKE/BETWEEN 조건 활용

변별력이 좋은 필수 조건이 있는 상황(ex. 당일 등록 상품은 소수)에서 LIKE/BETWEEN 사용은 나쁘지 않음.

```sql
-- 인덱스 : [등록일시, 상품분류코드]
select * from 상품
where 등록일시 >= trunc(sysdate) -- 필수 조건 (당일 등록 상품)
and 상품분류코드 like :prd_cls_cd -- 옵션 조건
```

- 필수 조건 칼럼을 인덱스 선두에 두고 액세스 조건으로 사용하면, LIKE/BETWEEN이 필터 조건이어도 좋은 성능 낼 수 있음.

```sql
-- 인덱스 : [상품명, 상품분류코드]
select * from 상품
where 상품명 = prd_nm           -- 필수 조건
and 상품분류코드 like :prd_cls_cd  -- 옵션 조건
```

- 필수 조건이 '='이면 옵션 조건인 상품분류코드까지 인덱스 액세스 조건으로 사용 가능

필수 조건의 변별력이 좋지 않을 때

```sql
-- 인덱스 : [상품대분류코드, 상품코드]
select * from 상품
where 상품대분류코드 = prd_lcls_cd    -- 필수 조건
and 상품코드 like :prd_cd || '%'     -- 옵션 조건
```

- 상품대분류코드만으로 조회할 때는 table full scan이 유리
- 옵티마이저는 상품코드까지 입력할 때를 기준으로 index range scan을 선택
- 사용자가 상품코드를 입력하지 않으면 성능이 좋지 않음

LIKE/BETWEEN 사용하고자 할 때는 아래 네 가지 경우에 속하는지 점검

1. 인덱스 선두 칼럼
2. null 허용 칼럼
3. 숫자형 칼럼
4. 가변 길이 칼럼

------

#### 1. 인덱스 선두 칼럼

인덱스 선두 칼럼에 대한 옵션 조건을 LIKE/BETWEEN 연산자로 처리하면 안 됨

```sql
-- 인덱스 : [고객ID, 거래일자]
select * from 거래
where 고객ID like :cust_id || '%'
and 거래일자 between :dt1 and :dt2
```

- 사용자가 고객ID 값을 입력하면, 둘 다 범위검색이라서 인덱스 스캔 과정에서 약간의 비효율이 있지만, 고객ID가 변별력이 매우 좋기 때문에 비교적 빠르게 조회됨
- 사용자가 고객ID 값을 입력하지 않으면, 인덱스에서 모든 거래 데이터 스캔하면서 고래 일자 조건 필터링
- 옵션 처리를 위와 같이 LIKE/BETWEEN 했다면, 인덱스를 [거래일자, 고객ID]로 구성해야 함
  - 고객ID를 입력할 경우 생기는 비효율은 감수

---

#### 2. NULL 허용 칼럼

null 허용 칼럼에 대한 옵션 조건을 LIKE/BETWEEN 연산자로 처리하면 안 됨
그냥 오류 남.

```sql
select * from 거래
where 고객ID like '%'
and 거래일자 between :dt1 and :dt2
```

- 고객ID가 안 들어오면 쿼리는 위와 같음
- 거래일자 조건에 해당하는 모든 고객의 거래를 선택해야 하는데, 고객ID가 null 허용 칼럼이고, 실제로 null 값이 입력되어 있다면, 해당 데이터는 결과집합에서 누락됨
- between 조건 사용할 때도 칼럼 값이 null인 데이터는 결과집합에서 누락됨

---

#### 3. 숫자형 칼럼

숫자형이면서 인덱스 액세스 조건으로도 사용 가능한 칼럼에 대한 옵션 조건 처리를 LIKE 사용하면 안 됨.

```sql
-- 인덱스 : [거래일자, 고객ID]
select * from 거래
where 거래일자 = :trd_dt
and 고객ID like :cust_id || '%'
```

- :cust_id에 값을 입력하면 두 칼럼 모두 인덱스 액세스 조건으로 사용 됨
- 만약 고객ID가 숫자형 칼럼이면 like 때문에 문자열로 자동 형변환이 일어나서 고객ID가 필터 조건으로 사용됨
  - to_char(고객ID) like :cust_id \|\| '%'
- [고객ID, 거래일자]로 구성한 인덱스는 아예 사용 불가 (숫자에 대해서 문자열로 검색하니까)

---

#### 4. 가변 길이 칼럼

LIKE 옵션 조건에 사용할 때는 칼럼 값 길이가 고정적이어야 함.

- ‘아진'이라는 이름을 찾기 위해서 like '아진%'을 사용하면 '아진진' 이름도 끌려올라옴

칼럼 값 길이가 가변적일 때는 변수 값 길이가 같은 레코드만 조회하도록 조건절 추가해야 함

```sql
where 고객명 like :cust_nm || '%'
and length(고객명) = length(nvl(:cust_nm, 고객명))
```

------

### union all 활용

```sql
select * from 거래
where :cust_id is null
and 거래일자 between :dt1 and :dt2
union all
select * from 거래
where :cust_id is not null
and 고객ID = :cust_id
and 거래일자 between :dt1 and :dt2
```

```
Execution Plan
--------------------------------------------------------------------
0    SELECT STATEMENT Optimizer=ALL_ROWS
1    0    CONCATENATION
2    1        FILTER    -- :cust_id is null
3    2            TABLE ACCESS (BY LOCAL INDEX ROWID) OF '거래' (TABLE)
4    3                INDEX (RANGE SCAN) OF '거래_IDX1' (INDEX)    -- 거래일자
5    1        FILTER    -- :cust_id is not null
6    5            TABLE ACCESS (BY LOCAL INDEX ROWID) OF '거래' (TABLE)
7    6                INDEX (RANGE SCAN) OF '거래_IDX2' (INDEX)    -- 고객ID + 거래일자
```

- :cust_id 에 값이 입력되지 않으면 거래일자가 선두인 인덱스를 사용하고
- 변수에 값을 넣으면 [고객ID, 거래일자] 인덱스 사용
- LIKE도 인덱스 사용은 가능하지만 필수 조건인 거래일자가 BETWEEN이면 옵션 조건 칼럼을 필터 조건으로 사용 (인덱스 후행 칼럼에서 LIKE를 필터로 사용)

```sql
-- [거래일자, 고객명]로 구성된 인덱스
SELECT * FROM 거래
WHERE 거래일자 BETWEEN :dt1 AND :dt2
AND 고객명 LIKE '홍길%'
```

- 거래일자는 BETWEEN 조건으로 인덱스 액세스 범위가 결정됨
- 인덱스의 후행 칼럼인 고객명에 LIKE 조건이 있다면, 이 LIKE 조건은 인덱스 필터 조건으로 사용됨



UNION ALL은 옵션 조건 칼럼도 인덱스 액세스 조건으로 사용 가능

---

### NVL/DECODE 함수 활용

```sql
select * from 거래
where 고객ID = nvl(:cust_id, 고객ID)
and 거래일자 between :dt1 and :dt2

select * from 거래
where 고객ID = decode(:cust_id, null, 고객ID, :cust_id)
and 거래일자 between :dt1 and :dt2
```

```
Execution Plan
--------------------------------------------------------------------
0    SELECT STATEMENT Optimizer=ALL_ROWS
1    0    UNION ALL
2    1        FILTER    -- :cust_id is null
3    2            TABLE ACCESS (BY LOCAL INDEX ROWID) OF '거래' (TABLE)
4    3                INDEX (RANGE SCAN) OF '거래_IDX1' (INDEX)    -- 거래일자
5    1        FILTER    -- :cust_id is not null
6    5            TABLE ACCESS (BY LOCAL INDEX ROWID) OF '거래' (TABLE)
7    6                INDEX (RANGE SCAN) OF '거래_IDX2' (INDEX)    -- 고객ID + 거래일자
```

- :cust_id 입력하지 않으면 거래일자가 선두 칼럼인 인덱스 사용
- 입력하면 [고객ID, 거래일자] 인덱스 사용
- 고객ID 칼럼을 함수 인자로 사용했는데도 인덱스를 사용 가능 (인덱스 칼럼을 가공)
  - OR Expansion 쿼리 변환이 일어났기 때문
  - UNION ALL 방식으로 옵티마이저가 쿼리를 변환
  - 이 기능이 작동하지 않으면 nvl, decode 사용하는 패턴도 인덱스 액세스 조건으로 사용 불가
  - :cust_id 입력하지 않으면 조건절이 '고객ID =  고객ID' 가 돼 버려서 시작점 찾을 수 없음
- LIKE과 같이, null 허용 칼럼에 대해서 사용 불가
- 조건절 변수에 null 입력하면 값이 null인 레코드가 결과집합에서 누락되기 때문
- 옵션 조건 처리 용으로 NVL/DECODE 함수 여러 개 사용하면 그 중에 변별력이 가장 좋은 칼럼 기준으로 한 번만 OR Expansion 일어남
  - OR Expansion 기준으로 선택되지 않으면 인덱스 구성 칼럼이어도 모두 필터 조건으로 처리됨

---

## 함수호출 부하 해소를 위한 인덱스 구성

PL/SQL의 사용자 정의 함수는 매우 느림

- VM 상에서 실행되는 인터프리터 언어
- 호출 시마다 context switching 발생
- 내장 SQL에 대한 recursive call 발생

PL/SQL로 작성함 함수와 프로시저를 컴파일하면 바이트코드를 생성해서 데이터 딕셔너리에 저장함. 이걸 해석할 수 있는 PL/SQL 엔진만 있으면 어디서든 실행할 수 있음.

PL/SQL 엔진은 바이트코드를 런타임 시 해석하면서 실행. JAVA처럼 인터프리터 언어라서 Native 코드로 완전 컴파일된 내장 함수에 비해 많이 느림.

PL/SQL 함수는 실행 시 매번 SQL 실행엔진과 PL/SQL 가상머신 사이에서 컨텍스트 스위칭 일어남.

- 함수를 작은 단위로 모듈화, 공용화 하면 느려짐

```sql
select 회원번호, 회원명, 생년, 생월일, GET_ADDR(우편번호) as 기본주소
from 회원
where 생월일 like '01%'
```

- 조건을 만족하는 회원 100만 명 있으면 GET_ADDR 함수도 100만 번 실행
- 함수에 SQL이 내장돼 있으면 그 SQL도 100만 번 실행
- recursive call 부하가 큼

```sql
select a.회원번호, a.회원명, a.생년, a.생월일,
			(select b.시도 || ' ' || b.구군 || ' ' || b.읍면동
       from 기본주소 b
       where b.우편번호 = a.우편번호
       and b.순번 = 1) 기본주소
from 회원 a
where a.생월일 like '01%'
```

- 이렇게 SQL만을 이용하는 게 훨씬 빠름

### 함수호출 최소화

그럼에도 PL/SQL을 써야만 하는 상황이 있음. 로직이 복잡할 때.

인덱스를 통해서 함수 호출을 최소화할 수 있음.

```sql
select /*+ full(a) */ 회원번호, 회원명, 생년, 생월일, 등록일자
from 회원
where 생년 = '1987'
  and 암호화된_전화번호 = encryption(:phone_no)
```

- encryption 함수는 조건절을 만족하는 건수만큼 수행됨

```sql
create index 회원_X01 on 회원(생년);
create index 회원_X02 on 회원(생년, 생월일, 암호화된_전화번호);
create index 회원_X03 on 회원(생년, 암호화된_전화번호);
```

- 회원_X01 인덱스 사용하면
  - 암호화된_전화번호 조건절을 테이블 액세스 단계에서 필터링
  - encryption 함수는 테이블 액세스 횟수만큼 수행 = '생년 = '1987'' 을 만족하는 건수만큼
- 회원_X02 인덱스 사용하면
  - 선행칼럼인 생월일에 대한 '=' 조건절이 없으므로 '암호화된_전화번호'는 인덱스 필터 조건
  - encryption 함수는 인덱스 스캔 횟수만큼 수행 = '생년 = '1987'' 을 만족하는 건수만큼
- 회원_X03 인덱스 사용하면
  - '암호화된_전화번호'도 인덱스 액세스 조건으로 사용
  - encryption 함수는 한 벚 수행

----

출처

친절한 SQL 튜닝