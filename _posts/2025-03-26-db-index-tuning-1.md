---
title: "[DB] 인덱스 튜닝 - 테이블 액세스 최소화"
date: 2025-03-26
categories: ["Database"]
tags: ["index", "index-tuning", "table access", ""]
use_math: true
---


## table random access

### index ROWID

찾고자 하는 칼럼이 모두 인덱스를 구성하고 있는 게 아니라면, 인덱스에서 ROWID를 찾아서 원본 테이블을 엑세스 해야 함.

ROWID는 데이터 파일 번호, 블록 번호, 로우 번호로 이루어져 있어 물리적 주소로 보일 수도 있지만, 논리적 주소. 물리적으로 직접 연결되지 않고, 테이블 레코드를 찾아가기 위한 논리적 주소 정보를 담고 있음 (7번 데이터파일, 123번 블록, 10번째 레코드).

ROWID는 포인터와는 다르다. (바로 주소 값으로 가면 값이 있는 게 아님)

### In-Memory DB (IMDB)

Main Memory DB (MMDB)라고도 부르는 데이터베이스는 데이터를 모두 메모리에 로드해 놓고, 메모리를 통해서만 I/O를 수행함.

OLTP 데이터베이스도 잘 튜닝되어 있다면 캐시 히트율이 99%이상 나오는데, IMDB만큼 빠르지는 않음. 대량 데이터를 액세스할 때는 차이가 더 많이 남.

IMDB

1. 인스턴스 기동 시 디스크에 저장된 데이터를 버퍼캐시로 로드
2. 인덱스 생성
   - 디스크 상의 주소정보가 아니라 메모리 상 주소정보를 포인터로 가짐

오라클은 테이블 블록이 계속 버퍼캐시에 캐싱됐다가 밀려났다가 다시 캐싱되길 반복. 캐싱 될 때마다 다른 곳에 적재되기 때문에 일정한 주소값을 갖지 않음.

$\therefore$ 메모리 주소가 아닌 디스크 주소 정보를 이용해 해시 알고리즘으로 버퍼 블록 찾아감

---



### I/O 메커니즘

DBA(Data Block Address, 데이터파일 번호 + 블록 번호)는 디스크에서 블록을 찾기 위한 주소.

블록 읽을 때는 디스크로 바로 가기 전에 버퍼캐시 확인.

- 캐시에 적재할 때와 읽을 때 같은 해시 함수 사용 $\rightarrow$ 버퍼 헤더는 항상 같은 해시 체인에 위치

- 실제 데이터가 담긴 버퍼 블록은 매번 다른 위치에 적재

- 버퍼 헤더는 버퍼 블록의 메모리 주소값을 갖고 있음

  

$\therefore$ 버퍼 캐시에서 블록을 검색하는 과정

1. 해싱 알고리즘을 사용하여 버퍼 헤더 찾아
2. 해당 버퍼 헤더를 시작점으로 버퍼 체인을 따라가
3. 원하는 블록을 만나면
4. 그 블록에서 얻은 포인터를 사용해서 실제 버퍼 블록의 위치로 찾아가

  

인덱스로 테이블 목록 액세스할 때는 리프 블록에서 읽은 ROWID를 분해하여 DBA 정보 추출.

테이블 full scan 할 때는 익스텐트 맵을 통해 읽을 블록들의 DBA 정보를 얻음.

  

모든 블록이 캐싱돼 있어도 매번 DBA 해싱과 래치 획득을 반복해야 함. 동시 액세스가 심할 때는 캐시버퍼 체인 래치와 버퍼 lock에 대한 경합이 발생함.

ROWID는 생각보다 고비용이다.

---

## 인덱스 클러스터링 팩터

Clusterfing FActor(CF) : 특정 칼럼을 기준으로 같은 값을 갖는 데이터가 서로 모여있는 정도

CF가 좋은 칼럼에 생성한 인덱스는 검색 효율이 좋음. *거주지역='제주'* 에 해당하는 고객 데이터가 물리적으로 근접해 있으면 데이터를 찾는 속도가 빠름.

![Image](https://github.com/user-attachments/assets/36083bff-3226-4bf1-8dec-344e0b52f19e)

![Image](https://github.com/user-attachments/assets/4ca68563-068b-469d-9a51-acee51665343)

위 두 개의 인덱스를 비교하면 인덱스 클러스터링 팩터가 높고 낮은 것을 비교할 수 있음.

CF가 좋으면 테이블 액세스량에 비해 블록 I/O가 적게 발생함.

**버퍼 pinning** : 오라클은 래치 획득, 해시 체인 스캔을 통해 얻은 테이블 블록에 대한 포인터를 바로 해제하지 않고 유지함.
그 다음 인덱스 레코드를 읽었는데, 바로 직전과 같은 테이블 블록을 가리키면, 유지하고 있는 포인터를 사용하여 바로 테이블 블록을 읽을 수 있음.
$\rightarrow$ 논리적 블록 I/O 생략

![Image](https://github.com/user-attachments/assets/d4e1de5c-c7b3-48c0-85f4-663a800de321)

실선에서 실제 블록 I/O가 발생하고, 점선에서 저장된 포인터를 사용함.

CF가 안 좋은 인덱스를 사용하면 테이블을 액세스 하는 횟수만큼 블록 I/O가 발생.

---

## 인덱스 손익분기점

![Image](https://github.com/user-attachments/assets/9b3cd99f-e814-423f-9220-e0d3f9621552)

위에서도 봤듯이, 인덱스 ROWID를 이용한 테이블 액세스는 고비용.

읽어야 할 테이터가 일정량을 넘으면 테이블 전체를 스캔하는 것보다 오히려 느려짐. Full table scan보다 느려지는 지점을 '인덱스 손익분기점'이라고 부름.

Full table scan은 성능이 일정함. select를 레코드 몇 개를 하든, 어차피 테이블은 다 읽을 거.

인덱스를 이용해서 테이블 액세스할 때는 몇 건을 추출하냐에 따라 성능 달라짐.

인덱스를 이용한 테이블 액세스가 느려지는 가장 핵심적인 두 가지 요인:

1. table full scan은 sequential access, 인덱스 ROWID를 이용한 테이블 액세스는 random access
2. table full scan은 multiblock I/O, 인덱스 ROWID를 이용한 테이블 액세스는 single block I/O

보통 인덱스 손익분기점은 5~20% 수준에서 잡힘.

그리고 CF에 따라서도 달라짐. CF 나쁘면 손익분기점이 1%까지도 갈 수 있음. CF 좋으면 90%까지도 감.

  

테이블 크기가 커지면 손익분기점은 더 낮아짐.

10만 건 테이블에서 10% = 만 건

- 버퍼캐시에서 데이터를 찾을 가능성이 있음
- 인덱스 칼럼 기준으로 값이 같은 테이블 레코드가 근처에 모여 있을 가능성 있음
- 계속 액세스하다 보면 언젠가부터는 계속 블록을 캐시에서만 가져오게 됨

1000만 건 테이블에서 10% = 100만 건

- 인덱스로 추출하면 정말 느려짐
- 조회 건수가 늘어날 수록 버퍼캐시에서 찾을 가능성이 낮아짐

버퍼캐시에 보통 데이터베이스에 저장된 전체 테이블에 대해서 수백만 개 블록 캐싱.
특정 테이블을 인덱스로 100만 건 이상 액세스하면 캐시 히트율 굉장히 낮음.

테이블 커지면 인덱스 칼럼 기준으로 값이 같은 레코드가 모여 있을 가능성(CF)이 매우 작아서 결국 모든 블록을 디스크에서 읽게 됨.

만 건만 넘어도 sequential access + multiblock I/O (= table full scan)가 더 빠를 수도.

---

### 온라인 프로그램 튜닝 vs. 배치 프로그램 튜닝

#### 온라인 프로그램

- 소량 데이터를 읽고 갱신
- 인덱스를 효과적으로 활용하는 것이 중요
- 조인은 대부분 NL 사용 (indexed).
  - 인덱스를 이용해 소트 연산을 생략할 수 있다면 온라인 환경에서 대량 데이터를 조회할 때 빠른 응답 속도 낼 수 있음

#### batch 프로그램

- 대량 데이터를 읽고 갱신
- 항상 전체범위 처리 기준으로 튜닝
- Index + NL join < Full scan + hash join

```sql
select c.고객번호, c.고객명, h.전화번호, h.주소, h.상태코드, h.변경일시
from 고객 c, 고객변경이력 h
where c.실명확인번호 = :rmnno
  and h.고객번호 = c.고객번호
  and h.변경일시 = (select max(변경일시)
                  from 고객변경이력 m
               		where 고객번호 = c.고객번호
               			and 변경일시 >= trunc(add_months(sysdate, -12), 'mm')
               			and 변경일시 < trunc(sysdate, 'mm'))
```

```txt
Execution Plan
------------------------------------------------------------------
0        SELECT STATEMENT Optimizer=ALL_ROWS (Cost=19.12)
1     0  NESTED LOOPS (Cost=19.12)
2     1    NESTED LOOPS (Cost=14.85)
3     2      NESTED LOOPS (Cost=9.26)
4     3        TABLE ACCESS (BY INDEX ROWID) OF '고객' (TABLE) (Cost=3.47)
5     4          INDEX (RANGE SCAN) OF '고객_X01' (INDEX) (Cost=1.21)
6     3      VIEW PUSHED PREDICATE OF 'SYS.VW_SQ_1' (VIEW) (Cost=2.64)
7     6        SORT (AGGREGATE) (Cost=2.64)
8     7          FIRST ROW (Cost=2.13)
9     8            INDEX (RANGE SCAN (MIN/MAX)) OF '고객변경이력_PK' (Cost=1.87)
10    2      INDEX (UNIQUE SCAN) OF '고객변경이력_PK' (INDEX (UNIQUE)) (Cost=0.89)
11    1    TABLE ACCESS (BY INDEX ROWID) OF '고객변경이력' (TABLE) (Cost=4.27)
```

- 실명확인번호로 조회한 특정 고객의 '최근 1년 이내 변경 이력 중 전월 말일 데이터' 출력
- 실명확인번호 조건에 해당하는 데이터는 한 건이거나 소량
  - index + NL join 사용이 효과적

```sql
insert into 고객_임시
select c.고객번호, c.고객명, h.전화번호, h.주소, h.상태코드, h.변경일시
from 고객 c, 고객변경이력 h
where c.고객구분코드 = 'A001'
  and h.고객번호 = c.고객번호
  and h.변경일시 = (select max(변경일시)
                  from 고객변경이력 m
               		where 고객번호 = c.고객번호
               			and 변경일시 >= trunc(add_months(sysdate, -12), 'mm')
               			and 변경일시 < trunc(sysdate, 'mm'))
```

- 고객구분코드가 'A001'인 고객의 최근 1년 이내 변경 이력 중 전월 말일 데이터를 읽어 고객_임시 테이블에 입력
- 전체 고객이 300만 건이고, 고객구분코드 조건을 만족하는 고객이 100만 명이라면 위와 같이 입력하는 건 성능이 좋지 않음

```sql
insert into 고객_임시
select /*+ full(c) full(h) index_ffs(m.고객변경이력)
				   ordered no_merge(m) use_hash(m) use_hash(h) */
				c.고객번호, c.고객명, h.전화번호, h.주소, h.상태코드, h.변경일시
from 고객 c,
		 (select 고객변호, max(변경일시) 최종변경일시
     	from 고객변경이력
      where 변경일시 >= trunc(add_months(sysdate, -12), 'mm')
        and 변경일시 < trunc(sysdate, 'mm')
      group by 고객번호) m,
     고객변경이력 h
where c.고객구분코드 = 'A001'
  and m.고객번호 = c.고객번호
  and h.고객번호 = m.고객번호
  and h.변경일시 = m.최종변경일시
```

```txt
Execution Plan
------------------------------------------------------------------
0        INSERT STATEMENT Optimizer=ALL_ROWS (Cost=3457865.34)
1     0    LOAD TABLE CONVENTIONAL OF '고객_임시' 
2     1      HASH JOIN (Cost=2854976.28)
3     2        HASH JOIN (Cost=1987543.67)
4     3          TABLE ACCESS (FULL) OF '고객' (TABLE) (Cost=754321.56)
5     3          VIEW (Cost=645832.23)
6     5            SORT (GROUP BY) (Cost=598752.47)
7     6              FILTER
8     7                INDEX (FAST FULL SCAN) OF '고객변경이력_PK' (Cost=387654.32)
9     2        TABLE ACCESS (FULL) OF '고객변경이력' (TABLE) (Cost=867432.89)
```

- 이렇게 full table scan + hash join 사용

고객변경이력 테이블 두 번 읽는 비효율 없애려면 윈도우 함수 사용

```sql
insert into 고객_임시
select 고객번호, 고객명, 전화번호, 주소, 상태코드, 변경일시
from (select /*+ full(c) full(h) leading(c) use_hash(h) */
     				 c.고객번호, c.고객명, h.전화번호, h.주소, h.상태코드, h.변경일시,
     				 rank() over (partition by h.고객번호 order by h.변경일시 desc) no
      from 고객 c, 고객변경이력 h
      where c.고객구분코드 = 'A001'
        and h.변경일시 >= trunc(add_months(sysdate, -12), 'mm')
        and h.변경일시 < trunc(sysdate, 'mm')
        and h.고객번호 = c.고객번호)
where no = 1
```

```txt
Execution Plan
------------------------------------------------------------------
0        INSERT STATEMENT Optimizer=ALL_ROWS (Cost=3548965.28)
1     0    LOAD TABLE CONVENTIONAL OF '고객_임시'
2     1      VIEW (Cost=2965432.17)
3     2        WINDOW (SORT PUSHED RANK) (Cost=2654321.93)
4     3          FILTER (Cost=1987654.65)
5     4            HASH JOIN (Cost=1654321.78)
6     5              TABLE ACCESS (FULL) OF '고객' (TABLE) (Cost=754321.56)
7     5              TABLE ACCESS (FULL) OF '고객변경이력' (TABLE) (Cost=867432.89)
```

대량 배치 프로그램에서는 인덱스보다 full scan이 효과적이지만, 초대용량 테이블을 full scan 하는 거 기다리는 것도 부담.

- 배치 프로그램에서는 파티션 활용이 매우 중요한 튜닝 요소

  - 위 쿼리의 고객변경이력 테이블을 변경일시 기준으로 파티셔닝하면, 변경일시 조건에 해당하는 파티션만 골라서 full scan

- 병렬 처리까지 할 수 있으면 더 좋음

  

파티션 테이블에도 인덱스를 사용할 수 있지만, 월 단위로 파티션한 테이블에서 특정 월 또는 몇 개월 치 데이터를 조회할 때 인덱스는 좋은 선택이 아님

- 보름/일주일 치 데이터를 조회하더라도 인덱스보다 full scan이 유리

- 2-3일 데이터 조회할 때도 full scan이 유리할 수도 있음



테이블을 파티셔닝하는 이유는 결국 full scan을 빠르게 처리하기 위해서

- 모든 성능 문제를 인덱스로 해결하려 하면 안 됨.

- 인덱스는 여러 튜닝 도구 중 하나일 뿐, 큰 테이블에서 아주 적은 일부 데이터를 빨리 찾고자 할 때 주로 사용



---

## 인덱스 칼럼 추가

테이블 액세스 최소화를 위해 가장 일반적으로 사용하는 기법은 인덱스에 칼럼 추가

EMP 테이블) PK + [DEPTNO, JOB]로 구성된 인덱스 EMP_X01 있을 때

```sql
select /*+ index(emp emp_x01) */ *
from emp
where deptno = 30
  and sal >= 2000
```

![Image](https://github.com/user-attachments/assets/30130a1b-b3be-4e45-bd59-112403d75973)

- 조건을 만족하는 사원은 한 명인데 테이블 6번 액세스
- 인덱스 구성을 [DEPTNO, SAL] 순으로 변경하면 좋겠지만, 현재 상태의 인덱스를 사용하는 쿼리가 있을 수 있기 때문에 인덱스 구성을 변경하기 쉽지 않음
- 그렇다고 인덱스를 계속 생성하면 인덱스 관리 비용이 증가하고 DML 부하에 따른 트랜잭션 성능 저하가 생길 수 있음
- 이럴 때 인덱스에 칼럼을 추가해 줄 수 있음
- 인덱스 스캔량은 줄지 않지만, 테이블 랜덤 액세스 횟수를 줄여줌

![Image](https://github.com/user-attachments/assets/cb7126f5-c484-408c-9b7e-157794c37302)

```sql
select 렌탈관리번호, 고객명, 서비스관리번호, 서비스번호
from 로밍렌탈
where 서비스번호 like '010%'
  and 사용여부 = 'Y'
```

위 쿼리에 대해서 '서비스번호' 단일 칼럼으로 구성된 인덱스를 사용한 결과가 아래와 같았다고 한다.

```txt
Call     Count CPU Time Elapsed Time     Disk      Query     Current      Rows
------- ------ -------- ------------- ---------- ---------- ---------- ----------
Parse        1    0.010        0.012          0          0          0          0
Execute      1    0.000        0.000          0          0          0          0
Fetch       78   10.150       49.199      27830     266968          0       1909
------- ------ -------- ------------- ---------- ---------- ---------- ----------
Total       80   10.160       49.211      27830     266968          0       1909

Rows    Row Source Operation
------- -------------------------------------------------------------------
  1909  TABLE ACCESS BY INDEX ROWID 로밍렌탈 (cr=266968 pr=27830 pw=0 time= ... )
266476  INDEX RANGE SCAN 로밍렌탈_N2 (cr=1011 pr=900 pw=0 time=1893462 us)
```

- 인덱스 스캔 266,476건에 대해서 테이블 랜덤 액세스를 265,957개의 블록을 읽음
- clustering factor이 매우 안 좋음
- 최종 결과 집합은 1,909건밖에 안 됨
  - *사용여부 = 'Y'* 에서 다 걸러짐

인덱스에 사용여부 칼럼을 추가하고 나면 트레이스가 아래와 같음

```txt
Call     Count CPU Time Elapsed Time     Disk      Query     Current      Rows
------- ------ -------- ------------- ---------- ---------- ---------- ----------
Parse        1    0.000        0.001          0          0          0          0
Execute      1    0.000        0.000          0          0          0          0
Fetch       78    0.140        0.154          0       2902          0       1909
------- ------ -------- ------------- ---------- ---------- ---------- ----------
Total       80    0.140        0.156          0       2902          0       1909

Rows    Row Source Operation
------- -------------------------------------------------------------------
  1909  TABLE ACCESS BY INDEX ROWID 로밍렌탈 (cr=2902 pr=0 pw=0 time= ... )
  1909  INDEX RANGE SCAN 로밍렌탈_N2 (cr=1001 pr=0 pw=0 time=198557 us)
```

인덱스에서 읽고 테이블 액세스를 했는데 버려지는 레코드가 많을 때 인덱스에 칼럼을 추가하면 성능 향상



---

## 인덱스만 읽고 처리

만약 테이블 랜덤 액세스가 많지만 버려지는 레코드가 거의 없다면?

```sql
select 부서번호, SUM(수량)
from 판매집계
where 부서번호 like '12%'
group by 부서번호;
```

- 부서번호 단일 칼럼으로 구성된 인덱스를 사용할 때, 비효율은 없음
- 인덱스에서 찾아서, 테이블 액세스 했다가 버리는 레코드가 없음

한 가지 시도해 볼 수 있는 건 Covered Query, Covered Index

Covered Query : 인덱스만 읽어서 처리하는 쿼리

Covered Index : Covered Query에서 사용한 인덱스

부서번호로 구성된 인덱스에 수량 칼럼을 추가하면 테이블 액세스를 제거할 수 있음

---

### include index

오라클에는 없지만 SQL Server에는 있음.

인덱스 키 외에 미리 지정한 칼럼을 리프 레벨에 함께 저장

```sql
create index emp_x01 on emp (deptno) include (sal)
```

- SAL 칼럼을 리프 블록에만 저장
- 수직적 탐색에는 DEPTNO만 사용
- 수평적 탐색에는 SAL 칼럼도 필터 조건으로 사용 가능 (table random access 횟수 줄이는 용도로만 사용)

```sql
create index emp_x02 on emp (deptno, sal)
```

- DEPTNO와 SAL 칼럼을 루트와 브랜치 믈록에 모두 저장
- 수직적 탐색에 사용 가능

```sql
select sal from emp where deptno = 20
```

- EMP_X01, EMP_X02 둘 다 covered index
  - table random access 생략

```sql
select * from emp where deptno = 20 and sal >= 2000
select * from emp where deptno = 20 and sal <= 3000
select * from emp where deptno = 20 and sal between 2000 and 3000
```

- 위 쿼리를 처리할 때 테이블 랜덤 액세스 측면에서는 작업량 똑같음
- 인덱스 스캔량은 EMP_X02가 더 적음

```sql
select sal from emp where deptno = 20 order by sal
```

- EMP_X02는 소트 연산 생략 가능

- EMP_X01는 소트 연산 생략 불가능

  

그럼 include index는 뭐가 좋냐?

- 인덱스 크기 최적화
- 인덱스 유지 비용 감소
  - include 칼럼은 인덱스 키의 일부가 아니기 때문에 정렬되지 않음
  - include 칼럼 값 바뀌어도 인덱스 재구성 덜 발생
- 인덱스에 키로 사용할 수 없는 대형 데이터 타입을 include 칼럼으로 추가할 수도
- 필터링과 정렬에 필요한 키 칼럼만 인덱스 키로 유지하면서, 조회에 필요한 추가 칼럼을 include로 추가

```sql
select ename, sal from emp where deptno = 20
```

- (deptno, sal) 인덱스보다 (deptno) + include(sal)이 더 효과적일 수 있음



----

## 인덱스 구조 테이블

인덱스를 이용한 테이블 액세스가 고비용이라면

랜덤 액세스가 아예 발생하지 않도록, 테이블을 인덱스 구조로 생성하면?

오라클에서는 IOT(Index-Organized-Table), SQL Server에서는 Clustered Index라고 부름.

- 일반 인덱스는 리프 블록에 테이블을 찾아가기 위한 ROWID를 저장
- IOT는 그 자리에 테이블 데이터를 저장
  - 테이블 블록에 있어야 할 데이터를 인덱스 리프 블록에 모두 저장

```sql
create table index_org_t (a number, b varchar(10),
                          constraint index_org_t_pk primary key (a))
organization index;
```

- 인덱스 구조로 만들겠다

- 일반 테이블은 힙 구조 테이블

  - 테이블 생성할 때 대개 생략하지만 명시할 수도 있음

  - ```sql
    create table index_org_t (a number, b varchar(10),
                              constraint index_org_t_pk primary key (a))
    organization heap;
    ```

  

- 일반 힙 구조 테이블에 데이터를 입력할 때는 랜덤 방식 사용

  - Freelist로부터 할당받은 블록에 정해진 순서 없이 데이터 입력

- IOT는 정렬 상태를 유지하며 데이터 입력

  

IOT는 인위적으로 clustering factor를 좋게 만드는 방법 중 하나

- 같은 값을 가진 레코드들이 정렬된 상태로 모여 있으므로 랜덤 액세스가 아닌 sequential 방식으로 액세스

  - between이나 부등호 조건으로 넓은 범위 읽을 때 유리

  

데이터 입력과 조회 패턴이 서로 다른 테이블에도 유용

- 영업사원 100명에 대한 일별 실적을 집계하는 테이블 (한 블록에 레코드 100개 저장)
  - 매일 한 블록씩 1년이면 365개 블록
- 실적등록은 일자별로 진행되지만, 실적조회는 사원별로 이루어짐
- 인덱스를 사용하면 사원마다 랜덤 액세스 방식으로 365개 테이블 블록 읽어야 함
  - 조회 건수만큼 블록 I/O
- 사번이 첫 번째 정렬 기준이 되도록 IOT를 구성하면 4개 블록만 읽고(100*4=400>365) 처리할 수 있음



---

## 클러스터 테이블

### 인덱스 클러스터 테이블

인덱스 클러스터 테이블은 클러스터 키 값이 같은 레코드를 한 블록에 모아서 저장 (물리적으로 같은 데이터 블록에 관련 데이터 저장).

한 블록에 모두 담을 수 없으면 새로운 블록 할당받아서 클러스터 체인으로 연결

![Image](https://github.com/user-attachments/assets/183a0479-d777-4388-a178-1b7ea87290d2)

다중 테이블 클러스터 : 여러 테이블 레코드를 같은 블록에 저장할 수도 있음

- 일반 테이블은 하나의 데이터 블록을 여러 테이블이 공유할 수 없음.

MS SQL Server의 Clustered Index랑 다른 거. Clustered Index는 IOT에 가가움.

오라클 클러스터는 키 값이 가능 데이터를 같은 공간에 저장만 하고, 클러스터형 인덱스처럼 정렬하지는 않음.

  

```sql
create cluster c_dept# (depno number(2)) index;
```

```sql
create index c_dept# on clusted c_dept#;
```

- 클러스터에 테이블을 담기 전에 클러스터 인덱스를 반드시 정의해야 함
- 클러스터 인덱스는 클러스터 키 값과 해당 키 값이 저장된 데이터 블록의 주소를 가리킴
- 클러스터 인덱스는 데이터 검색 용도로 사용할 뿐만 아니라 데이터가 저장될 위치를 찾을 때도 사용

클러스터 인덱스 만들었으면 클러스터 테이블 생성

```sql
create table dept (
	deptno number(2)   not null,
	dname varchar2(14) not null,
	loc varchar2(13))
cluster c_dept#(deptno);
```

![Image](https://github.com/user-attachments/assets/743dbfc9-1f87-4ff1-a2a6-8520ae3acfe2)

- 클러스터 인덱스도 일반 B*Tree 인덱스 구조 사용
- 테이블 레코드를 일일이 가리키지 않고 해당 값을 저장하는 첫 번째 데이터 블록을 가리킴
  - 일반 테이블에 생성한 인덱스 레코드는 테이블 레코드와 1:1 대응 관계를 가짐
  - 클러스터 인덱스는 테이블 레코드와 1:M 관계를 가짐
  - 클러스터 인덱스의 키값은 항상 unique.

- 클러스터 인덱스를 스캔하면서 값을 찾을 때 랜덤 액세스가 값 하나당 한 번씩 발생
  - 데이터가 너무 많아 하나의 블록에 저장되지 못하고 여러 블록에 걸쳐 클러스터 체인으로 연결된 경우에는 블록 수만큼 랜덤 액세스 하겠지
- 클러스터에 도달해서는 sequential 방식으로 스캔하기 때문에 넓은 범위를 읽어도 비효율이 없음

---

### 해시 클러스터 테이블

인덱스를 사용하지 않고 해시 알고리즘을 사용해 클러스터를 찾아감

![Image](https://github.com/user-attachments/assets/a3af872e-149c-4a1b-aa2d-9a9b963222fe)

클러스터 생성

```sql
create cluster c_dept# (deptno number(2)) hashkeys 4;
```

클러스터 테이블 생성

```sql
create table dept (
	deptno number(2)   not null,
	dname varchar2(14) not null,
	loc varchar2(13))
cluster c_dept#(deptno);
```

---

## 부분범위 처리

부분범위 처리를 활용하면 인덱스로 액세스할 대상 레코드가 많아도 아주 빠른 응답속도를 낼 수 있음.

DBMS가 클라이언트에게 데이터를 전송할 때, 일정량씩 나누어 전송.

전체 결과집합 중 먼저 읽는 데이터부터 일부(array size만큼)만 전송하고, 서버 프로세스는 CPU를 OS에 반환하고 클라이언트로부터 추가 fetch call이 오기 전까지 대기 큐에서 sleep.

다음 fetch call 받으면 그 다음 데이터부터 읽어서 전송하고 다시 자.

array size(데이터 전송 단위)는 클라이언트 프로그램에서 설정. 한번에 얼마나 필요한지에 따라서 그만큼만 받아오고, 필요하면 또 받아오고.

```java  
private void execute(Connection con) throws Exception {
    Statement stmt = con.createStatement();
    ResultSet rs = stmt.executeQuery("select name from big_table");

    for(int i=0; i<100; i++) {
        if(rs.next()) System.out.println(rs.getString(1));
    }

    rs.close();
    stmt.close();
}
```

- 예를 들어, 프론트엔드에서 데이터베이스 레코드 100개를 보여주는 페이지가 있어
- 처음 로드하면 첫 100개만 보여
- 사용자가 오른쪽 화살표 버튼 눌러서 101-200번 레코드 보고 싶어하면 또 그 다음 100개 요청해서 받아와
- 이렇게 하면 테이블에 레코드가 엄청 많아도 한 번에 다 읽어오지 않고 필요한 만큼씩 읽어오니까 처리 속도가 빠름

---

### 정렬 조건이 있을 때 부분범위 처리

```java
ResultSet rs = stmt.executeQuery("select name from big_table order by created");
```

만약에 쿼리에 order by가 있다면 전체범위 처리 해야 함.

다 읽고 created 순으로 정렬을 해야 클라이언트에게 데이터 전송을 시작할 수 있음.

created 칼럼이 선두 칼럼인 인덱스가 있으면 부분범위 처리 가능.

---

### 적정 Array Size 조정

전송해야 할 데이터량에 따라 Array Size 조정. Array Size를 늘리면 fetch call 횟수를 줄일 수 있음.

앞쪽 일부 데이터만 요청하는 프로그램에서는 Array Size 작게.

처음 100-200개 레코드만 필요한데 처음부터 1000개 읽어오면 네트워크, 서버, 클라이언트 자원 낭비.

---

### ## OLTP 환경에서 부분범위 처리

OLPT = Online Transaction Processing

- 주로 소량 데이터 처리
- 수천수만 건을 조회하려고 할 때 인덱스를 사용하면 랜덤 액세스가 많이 발생 (cache miss 많으면 엄청 오래 걸릴 수도)
- 한 번에 모든 데이터를 다 읽지 않고, 정렬 순서 상위 데이터만 확인하는 경우가 많음
  - 은행계좌 입출금 내역, 게시판 포스트 조회, etc.

```sql
select post_ID, title, author, date
from post
where typecode = 'A'
order by date desc
```

- 인덱스 선두 칼럼을 [typecode, date] 순으로 구성하지 않으면 소트 연산 생략 불가

```txt
| Id | Operation                     | Name        | Rows  | Bytes | Cost (%CPU)|
------------------------------------------------------------------
|  0 | SELECT STATEMENT              |             | 40000 | 3515K | 2041   (1) |
|  1 |  SORT ORDER BY                |             | 40000 | 3515K | 2041   (1) |
|  2 |   TABLE ACCESS BY INDEX ROWID | post        | 40000 | 3515K | 1210   (1) |
|* 3 |    INDEX RANGE SCAN           | post_X01    | 40000 |       | 96     (2) |
------------------------------------------------------------------

Predicate Information (identified by operation id):
---------------------------------------------------
2 - access("typecode"='A')
```

```txt
| Id | Operation                    | Name        | Rows  | Bytes | Cost (%CPU)|
------------------------------------------------------------------
|  0 | SELECT STATEMENT             |             | 40000 | 3515K | 1372   (1) |
|  1 | TABLE ACCESS BY INDEX ROWID  | post        | 40000 | 3515K | 1372   (1) |
|* 2 |  INDEX RANGE SCAN DESCENDING | post_X02    | 40000 |       | 258    (1) |
------------------------------------------------------------------

Predicate Information (identified by operation id):
---------------------------------------------------
2 - access("typecode"='A')
```

- 인덱스를 [typecode, date] 순으로 구성하면 SORT ORDER BY 생략 가능

---

## batch I/O

인덱스를 이용해 테이블을 액세스하다가, 버퍼 캐시에서 블록을 찾지 못하면 일반적으로 디스크 블록을 바로 읽음.

오라클은 12c 버전부터 batch I/O를 이용하여 테이블 블록에 대한 디스크 I/O call을 미뤘다가 읽을 블록이 일정량 쌓이면 한꺼번에 처리. (NL 조인에 사용되는 batch I/O랑은 별개)

### 데이터 정렬

batch I/O가 작동하면 인덱스를 이용해서 출력하는 데이터 정렬 순서가 매번 다를 수 있음.

테이블 블록을 모두 버퍼 캐시에서 찾을 때는 인데스 키값 순으로 데이터가 출력되지만, batch I/O가 작동하면 순서가 다를 수 있음.

```sql
create index emp_x01 on emp(deptno, job, empno);
set autotrace traceonly exp;
select * from emp e where deptno = 20 order by job, empno;
```

```txt
| Id | Operation                    | Name    | Rows | Bytes | Cost |
---------------------------------------------------------------------
|  0 | SELECT STATEMENT             |         |    5 |   190 |    2 |
|  1 |  TABLE ACCESS BY INDEX ROWID | EMP     |    5 |   190 |    2 |
|  2 |   INDEX RANGE SCAN           | EMP_X01 |    5 |       |    1 |
---------------------------------------------------------------------
```

- 인덱스를 이용해 소트 연산 생략

```sql
select /*+ batch_table_access_by_rowid(e) */ * 
from emp e 
where deptno = 20 
order by job, empno;
```

```txt
| Id | Operation                             | Name    | Rows | Bytes | Cost |
------------------------------------------------------------------------------
|  0 | SELECT STATEMENT                      |         |    5 |   190 |    2 |
|  1 |  SORT ORDER BY                        |         |    5 |   190 |    2 |
|  2 |   TABLE ACCESS BY INDEX ROWID BATCHED | EMP     |    5 |   190 |    2 |
|  3 |    INDEX RANGE SCAN                   | EMP_X01 |    5 |       |    1 |
------------------------------------------------------------------------------
```

- batch_table_access_by_rowid 힌트를 사용
- batch I/O가 작동할 수도 있다고 TABLE ACCESS에 BATCHED가 추가됨
- SORT ORDER BY 도 생략이 안 됐음
- batch I/O가 작동하면 데이터 정렬 순서를 보장할 수 없기 때문에 sort를 해야 함

```sql
select * from emp e where deptno = 20;
```

```txt
| Id | Operation                            | Name    | Rows | Bytes | Cost |
-----------------------------------------------------------------------------
|  0 | SELECT STATEMENT                     |         |    5 |   190 |    2 |
|  1 |  TABLE ACCESS BY INDEX ROWID BATCHED | EMP     |    5 |   190 |    2 |
|  2 |   INDEX RANGE SCAN                   | EMP_X01 |    5 |       |    1 |
-----------------------------------------------------------------------------
```

- 애초에 소트 연산을 생략할 수 없거나, 쿼리에 sort가 없으면 batch I/O를 사용하지 않을 이유가 없음

- 보통은 index, index_desc 힌트 써서 ORDER BY 생략하는 쿼리를 많이 썼었음.
- 하지만 12c로 업그레이드하면 정렬 순서가 달라질 수도 있어서 ORDER BY를 추가하는 것이 좋음

---

부분 범위 처리는 멈출 수 있어야 함.

요즘은 클라이언트와 DB 사이에 WAS, AP 서버가 있는 n-Tier 아키텍처가 대부분.

- 이런 경우 특정 DB 컨넥션 독점이 불가능. 컨넥션을 다시 connection pool에 반환해야 함.
- 조회 결과를 클라이언트에게 모두 전송하고 cursor을 닫아야 함.

n-Tier 아키텍처에서도 부분범위 처리는 여전히 가능

---

출처

친절한 SQL 튜닝

