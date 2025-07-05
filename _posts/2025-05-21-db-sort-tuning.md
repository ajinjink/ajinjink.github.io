---
title: "[DB] 소트 튜닝"
date: 2025-05-21
categories: ["Database"]
tags: ["sort-tuning", "index", "top n query", "predicate pushing", "sort area"]
use_math: true
---



SQL 수행 도중 가공된 데이터 집합이 필요할 때, PGA와 temp tablespace를 사용함.  
소트와 그룹핑 할 때도 동일.

<br/>

소트는 PGA에 할당한 Sort Area에서 이루어지고, 공간이 부족하면 디스크 temp table을 사용.

- in-memory sort (internal sort) : 전체 정렬 작업을 메모리 내에서 완료
- to-disk sort (external sort) : Sort Area 내에서 완료하지 못하고 디스크 공간까지 사용

<img src="/assets/images/2025-05-21/Picture1.png" alt="Picture1"/>

1. 소트 대상 집합을 SGA 버퍼캐시를 통해서 읽어
2. Sort Area에서 정렬 시도
3. Sort Area에서 정렬을 마무리하면 좋겠지만, 공간이 부족하다면
   정렬된 중간집합을 Temp 테이블스페이스에 임시 세그먼트를 만들어 저장
   - Sort Run : Sort Area가 찰 때마다 Temp 테이블스페이스에 저장해 둔 중간 단계의 집합

4. Sort Runs를 다시 Merge.
   어차피 각 Sort Run 내에서는 정렬이 되어 있어서 머지가 어렵진 않음.
   오름차순 정렬이면 가장 작은 값부터 PGA로 읽다가, PGA가 찰 때마다 쿼리 수행 다음 단계로 저장하거나 클라이언트에게 전송하면 됨.

<br/>

소트 연산은 Memory-intensive (메모리 집약적), CPU-intensive (CPU 집약적).

데이터량 많으면 디스크 I/O가 발생. 디스크 소트가 발생하는 수간 쿼리 성능은 나빠짐.

부분범위 처리가 불가능함 $\rightarrow$ OLTP 환경에서 애플리케이션 성능을 저하시키는 주요한 원인.

가능하면 소트가 발생하지 않도록 SQL을 작성하고, 소트가 불가피하다면 메모리 내에서 수행을 완료할 수 있도록 해야 함.



---



### 소트를 발생시키는 오퍼레이션

#### Sort Aggregate

Sort Aggregate는 전체 로우를 대상으로 한 집계를 수행할 때 나타남.

이름이 Sort지만 실제로 데이터를 정렬하진 않고, Sort Area를 사용하기만 함.

```sql
select sum(sal), max(sal), min(sal), avg(sal) from emp;
```

```
| Id | Operation          | Name | Rows | Bytes | Cost (%CPU)| Time     |
|----|--------------------|------|------|-------|------------|----------|
|  0 | SELECT STATEMENT   |      | 1    | 4     |    3 (0)   | 00:00:01 |
|  1 |  SORT AGGREGATE    |      | 1    | 4     |            |          |
|  2 |   TABLE ACCESS FULL| EMP  | 14   | 56    |    3 (0)   | 00:00:01 |
```

<img src="/assets/images/2025-05-21/Picture2.png" alt="Picture2"/>

데이터를 정렬하지 않고 SUM, MAX, MIN, AVG 구하는 순서

1. Sort Area에 SUM, MAX, MIN, COUNT 값을 위한 변수 하나씩 할당
2. 테이블에서 읽은 첫 번째 레코드 값을 SUM, MAX, MIN 변수에 저장하고, COUNT에는 1 저장
3. 두 번째 레코드부터 하나씩 읽어가면서 SUM 변수에는 값을 누적하고, MAX, MIN은 업데이트, COUNT는 1 증가시킴
4. SUM, MAX, MIN 은 저장되어 있는 값 출력하고, AVG는 SUM을 COUNT로 나눠서 출력


---

#### Sort Order By

Sort Order By는 데이터 정렬할 때 나타남.

```sql
select * from emp order by sal desc;
```

```
| Id | Operation          | Name | Rows | Bytes | Cost (%CPU)| Time     |
|----|--------------------|------|------|-------|------------|----------|
|  0 | SELECT STATEMENT   |      | 14   | 518   |  4 (25)    | 00:00:01 |
|  1 |  SORT ORDER BY     |      | 14   | 518   |  4 (25)    | 00:00:01 |
|  2 |   TABLE ACCESS FULL| EMP  | 14   | 518   |  3 (0)     | 00:00:01 |
```


---

#### Sort Group By

Sort Group By는 소팅 알고리즘을 사용해 그룹별 집계를 수행할 때 나타남.

```sql
select deptno, sum(sal), max(sal), min(sal), avg(sal)
from   emp
group by deptno
order by deptno ;
```

```
| Id | Operation          | Name | Rows | Bytes | Cost (%CPU)| Time     |
|----|--------------------|------|------|-------|------------|----------|
|  0 | SELECT STATEMENT   |      | 11   | 165   |  4 (25)    | 00:00:01 |
|  1 |  SORT GROUP BY     |      | 11   | 165   |  4 (25)    | 00:00:01 |
|  2 |   TABLE ACCESS FULL| EMP  | 14   | 210   |  3 (0)     | 00:00:01 |
```

부서번호 각각 sum, max, min, avg가 알고 싶음.

전체 사원의 급여 정보를 읽어서 부서번호 순으로 정렬하는 건 너무 비효율적.

<img src="/assets/images/2025-05-21/Picture3.png" alt="Picture3"/>

- 부서번호 별로 메모리를 할당하고, 부서번호 순으로 정렬.
  - 정렬되어 있어야 찾기 편함
- 레코드를 읽으면서, 각 부서번호 별로 SUM, MAX, MIN, COUNT 값 갱신
- 새로운 부서번호 만나면 새로운 SUM, MAX, MIN, COUNT 메모리 공간 할당

사원 수가 많아도 부서 개수만큼만 메모리를 쓰면 돼서 Sort Area가 클 필요가 없음.

집계할 대상 레코드 수가 많다고 Temp 테이블스페이스를 쓰는 건 아님.

---

group by 절 뒤에 order by 절을 명시하지 않으면 hash group by가 쓰인다.

```sql
select deptno, sum(sal), max(sal), min(sal), avg(sal)
from   emp
group by deptno ;
```

```
| Id | Operation           | Name | Rows | Bytes | Cost (%CPU)| Time     |
|----|---------------------|------|------|-------|------------|----------|
|  0 | SELECT STATEMENT    |      | 11   | 165   |  4 (25)    | 00:00:01 |
|  1 |  HASH GROUP BY      |      | 11   | 165   |  4 (25)    | 00:00:01 |
|  2 |   TABLE ACCESS FULL | EMP  | 14   | 210   |  3 (0)     | 00:00:01 |
```

Sort Group By에서는 메모리 공간을 찾기 위해 정렬을 했음.

Hash Group By는 소트 알고리즘 대신 해싱 알고리즘을 사용.

<img src="/assets/images/2025-05-21/Picture4.png" alt="Picture4"/>

레코드 읽을 때마다 Group By 칼럼의 해시 값으로 해시 버킷을 찾아 그룹별로 집계 항목 갱신.

---

#### Sort Unique

옵티마이저가 서브쿼리를 풀어서 일반 조인문으로 변환하는 게 subquery unnesting.

Unnesting된 서브쿼리가 M쪽 집합이면(1쪽 집합이더라도 조인 칼럼에 Unique 인덱스 없으면) 메인 쿼리와 조인하기 전에 중복 레코드 제거해야 함.

```sql
-- 고객(1)과 주문(M)의 관계
SELECT *
FROM 고객 c JOIN 주문 o ON c.고객ID = o.고객ID;
```

- 한 고객은 여러 건의 주문을 할 수 있음 $\rightarrow$ 고객이 1쪽, 주문이 M쪽 집합
- 고객ID는 고객 테이블에서 PK (unique), 주문 테이블에서는 FK (중복 가능)
- 서브쿼리가 M쪽이면 조인하면 중복이 생길 수 있으므로, 조인 전에 중복 제거해야 함.
  서브쿼리가 1쪽이면 중복이 없으므로 그대로 조인

```sql
select /*+ ordered use_nl(dept) */ * from dept
where deptno in (select /*+ unnest */ deptno 
                from emp where job = 'CLERK') ;
```

```
| Id | Operation                      | Name        | Rows | Bytes | Cost (%CPU)|
|----|--------------------------------|-------------|------|-------|------------|
|  0 | SELECT STATEMENT               |             | 3    | 87    |  4 (25)    |
|  1 |  NESTED LOOPS                  |             | 3    | 87    |  4 (25)    |
|  2 |   SORT UNIQUE                  |             | 3    | 33    |  2 (0)     |
|  3 |    TABLE ACCESS BY INDEX ROWID | EMP         | 3    | 33    |  2 (0)     |
|  4 |     INDEX RANGE SCAN           | EMP_JOB_IDX | 3    |       |  1 (0)     |
|  5 |   TABLE ACCESS BY INDEX ROWID  | DEPT        | 1    | 18    |  1 (0)     |
|  6 |    INDEX UNIQUE SCAN           | DEPT_PK     | 1    |       |  0 (0)     |
```

PK/Unique 제약 또는 Unique 인덱스를 통해 Unnesting된 서브쿼리의 유일성이 보장된다면 Sort Unique 오퍼레이션은 생략됨.

<br/>

Union, Minus, Intersect 같은 Set 연산자를 사용할 때도 Sort Unique 나타남.

```sql
select job, mgr from emp where deptno = 10
union
select job, mgr from emp where deptno = 20;
```

```
| Id | Operation           | Name | Rows | Bytes | Cost (%CPU)| Time     |
|----|---------------------|------|------|-------|------------|----------|
|  0 | SELECT STATEMENT    |      | 10   | 150   |  8 (63)    | 00:00:01 |
|  1 |  SORT UNIQUE        |      | 10   | 150   |  8 (63)    | 00:00:01 |
|  2 |   UNION-ALL         |      |      |       |           |          |
|  3 |    TABLE ACCESS FULL| EMP  | 5    | 75    |  3 (0)     | 00:00:01 |
|  4 |    TABLE ACCESS FULL| EMP  | 5    | 75    |  3 (0)     | 00:00:01 |
```

```sql
select job, mgr from emp where deptno = 10
minus
select job, mgr from emp where deptno = 20;
```

```
| Id | Operation           | Name | Rows | Bytes | Cost (%CPU)| Time     |
|----|---------------------|------|------|-------|------------|----------|
|  0 | SELECT STATEMENT    |      | 5    | 150   |  8 (63)    | 00:00:01 |
|  1 |  MINUS              |      |      |       |           |          |
|  2 |   SORT UNIQUE       |      | 5    | 75    |  4 (25)    | 00:00:01 |
|  3 |    TABLE ACCESS FULL| EMP  | 5    | 75    |  3 (0)     | 00:00:01 |
|  4 |   SORT UNIQUE       |      | 5    | 75    |  4 (25)    | 00:00:01 |
|  5 |    TABLE ACCESS FULL| EMP  | 5    | 75    |  3 (0)     | 00:00:01 |
```

Distinct 연산자 사용할 때도 Sort Unique 일어남.

```sql
select distinct deptno from emp order by deptno;
```

```
| Id | Operation           | Name | Rows | Bytes | Cost (%CPU)| Time     |
|----|---------------------|------|------|-------|------------|----------|
|  0 | SELECT STATEMENT    |      | 3    | 9     |  5 (40)    | 00:00:01 |
|  1 |  SORT UNIQUE        |      | 3    | 9     |  4 (25)    | 00:00:01 |
|  2 |   TABLE ACCESS FULL | EMP  | 14   | 42    |  3 (0)     | 00:00:01 |
```

10gR2부터는 distinct 연산에도 Hash Unique 방식 사용. Order By 생략하는 것도 Group By 생략할 때와 똑같음.

```sql
select distinct deptno from emp;
```

```
| Id | Operation           | Name | Rows | Bytes | Cost (%CPU)| Time     |
|----|---------------------|------|------|-------|------------|----------|
|  0 | SELECT STATEMENT    |      | 3    | 9     |  4 (25)    | 00:00:01 |
|  1 |  HASH UNIQUE        |      | 3    | 9     |  4 (25)    | 00:00:01 |
|  2 |   TABLE ACCESS FULL | EMP  | 14   | 42    |  3 (0)     | 00:00:01 |
```

---

#### Sort Join

Sort Join 오퍼레이션은 소트 머지 조인할 때 나타남.

```sql
select /*+ ordered use_merge(e) */ *
from   dept d, emp e
where  d.deptno = e.deptno ;
```

```
| Id | Operation           | Name | Rows | Bytes | Cost (%CPU)| Time     |
|----|---------------------|------|------|-------|------------|----------|
|  0 | SELECT STATEMENT    |      | 14   | 770   |  8 (25)    | 00:00:01 |
|  1 |  MERGE JOIN         |      | 14   | 770   |  8 (25)    | 00:00:01 |
|  2 |   SORT JOIN         |      | 4    | 72    |  4 (25)    | 00:00:01 |
|  3 |    TABLE ACCESS FULL| DEPT | 4    | 72    |  3 (0)     | 00:00:01 |
|  4 |   SORT JOIN         |      | 14   | 518   |  4 (25)    | 00:00:01 |
|  5 |    TABLE ACCESS FULL| EMP  | 14   | 518   |  3 (0)     | 00:00:01 |
```

---

#### Window Sort

Window Sort는 윈도우 함수를 수행할 때 나타남.

```sql
select empno, ename, job, mgr, sal, avg(sal) over (partition by deptno)
from emp;
```

```
| Id | Operation           | Name | Rows | Bytes | Cost (%CPU)| Time     |
|----|---------------------|------|------|-------|------------|----------|
|  0 | SELECT STATEMENT    |      | 14   | 406   |  4 (25)    | 00:00:01 |
|  1 |  WINDOW SORT        |      | 14   | 406   |  4 (25)    | 00:00:01 |
|  2 |   TABLE ACCESS FULL | EMP  | 14   | 406   |  3 (0)     | 00:00:01 |
```

---



## 소트가 발생하지 않도록 SQL 작성

### Union vs. Union All

Union을 사용하면 옵티마이저는 상단과 하단 두 집합 간 중복을 제거하려고 소트 작업을 수행.

Union All은 중복을 확인하지 않고 두 집합을 단순히 결합. 소트 작업 없음.

될 수 있으면 Union All을 사용하는 게 좋지만, 결과 집합이 동일하도록 신경써야 함.

| 결제번호 | 결제수단코드 | 주문번호 | 결제금액 | 결제일자 | 주문일자 |
| -------- | ------------ | -------- | -------- | -------- | -------- |
| 1        | M            | 2153     | 24,000   | ...      | ...      |
| 2        | M            | 3525     | 30,000   | ...      | ...      |
| 3        | M            | 5486     | 5,000    | ...      | ...      |
| 4        | C            | 5486     | 15,000   | ...      | ...      |
| 5        | C            | 8216     | 12,000   | ...      | ...      |
| 6        | C            | 8783     | 26,000   | ...      | ...      |

```sql
select 결제번호, 주문번호, 결제금액, 주문일자 ...
from   결제
where  결제수단코드 = 'M' and 결제일자 = '20180316'
UNION
select 결제번호, 주문번호, 결제금액, 주문일자 ...
from   결제
where  결제수단코드 = 'C' and 결제일자 = '20180316'
```

- 결제수단코드 조건절에 다른 값 입력했음
- 위 테이블에 대해서 위 쿼리는 Union 상단과 하단 집합 사이에 인스턴스 중복 가능성이 없음.

- Union을 사용함으로 인해 소트 연산 발생시킴

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=4 Card=2 Bytes=106)
1  0   SORT (UNIQUE) (Cost=4 Card=2 Bytes=106)
2  1     UNION-ALL
3  2       TABLE ACCESS (BY INDEX ROWID) OF '결제' (TABLE) (Cost=1 ... )
4  3         INDEX (RANGE SCAN) OF '결제_N1' (INDEX) (Cost=1 Card=1)
5  2       TABLE ACCESS (BY INDEX ROWID) OF '결제' (TABLE) (Cost=1 ... )
6  5         INDEX (RANGE SCAN) OF '결제_N1' (INDEX) (Cost=1 Card=1)
```

<img src="/assets/images/2025-05-21/Picture5.png" alt="Picture5" style="zoom:22%;" />

위 아래 두 집합이 상호배타적이므로 Union 대신 Union All 사용해도 됨.

---

```sql
select 결제번호, 결제수단코드, 주문번호, 결제금액, 결제일자, 주문일자 ...
from   결제
where  결제일자 = '20180316'
UNION
select 결제번호, 결제수단코드, 주문번호, 결제금액, 결제일자, 주문일자 ...
from   결제
where  주문일자 = '20180316'
```

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=2 Card=2 Bytes=106)
1  0   SORT (UNIQUE) (Cost=2 Card=2 Bytes=106)
2  1     UNION-ALL
3  2       TABLE ACCESS (BY INDEX ROWID) OF '결제' (TABLE) (Cost=0 ... )
4  3         INDEX (RANGE SCAN) OF '결제_N2' (INDEX) (Cost=0 Card=1)
5  2       TABLE ACCESS (BY INDEX ROWID) OF '결제' (TABLE) (Cost=0 ... )
6  5         INDEX (RANGE SCAN) OF '결제_N3' (INDEX) (Cost=0 Card=1)
```

- 이 쿼리는 상하단 집합 사이에 인스턴스 중복 가능성이 있음
  - 결제일자와 주문일자 조건은 상호배타적 조건이 아님

<img src="/assets/images/2025-05-21/Picture6.png" alt="Picture6" style="zoom:22%;" />

여기서 Union을 Union All로 바꾸면 결제일자와 주문일자가 같은 결제 데이터가 중복해서 출력됨.

소트 연산이 일어나지 않으면서 중복을 피하고 싶으면 아래와 같이 조건을 나눠줘야 함.

```sql
select 결제번호, 결제수단코드, 주문번호, 결제금액, 결제일자, 주문일자 ...
from   결제
where  결제일자 = '20180316'
UNION ALL
select 결제번호, 결제수단코드, 주문번호, 결제금액, 결제일자, 주문일자 ...
from   결제
where  주문일자 = '20180316'
and    결제일자 <> '20180316'
```

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=0 Card=2 Bytes=106)
1  0   UNION-ALL
2  1     TABLE ACCESS (BY INDEX ROWID) OF '결제' (TABLE) (Cost=0 Card=1 ... )
3  2       INDEX (RANGE SCAN) OF '결제_N2' (INDEX) (Cost=0 Card=1)
4  1     TABLE ACCESS (BY INDEX ROWID) OF '결제' (TABLE) (Cost=0 Card=1 ... )
5  4       INDEX (RANGE SCAN) OF '결제_N3' (INDEX) (Cost=0 Card=1)
```

만약에 결제일자가 Null 허용 칼럼이면 제일 마지막 조건을 아래와 같이 바꿔야 함.

```sql
and    (결제일자 <> '20180316' or 결제일자 is null)
```

```sql
and    LNNVL(결제일자 = '20180316')   -- null 데이터도 함께 조회되도록
```

---



### Exists

중복 레코드를 제거할 목적으로 Distinct 연산자를 사용하면, 조건에 해당하는 데이터를 모두 읽어서 중복을 제거해야 함.

부분범위 처리도 불가.

```sql
select DISTINCT p.상품번호, p.상품명, p.상품가격, ...
from   상품 p, 계약 c
where  p.상품유형코드 = :pclscd
and    c.상품번호 = p.상품번호
and    c.계약일자 between :dt1 and :dt2
and    c.계약구분코드 = :ctpcd
```

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=3 Card=1 Bytes=80)
1  0   HASH (UNIQUE) (Cost=3 Card=1 Bytes=80)
2  1     FILTER
3  2       NESTED LOOPS
4  3         NESTED LOOPS (Cost=2 Card=1 Bytes=80)
5  4           TABLE ACCESS (BY INDEX ROWID) OF '상품' (TABLE) (Cost=1 ... )
6  5             INDEX (RANGE SCAN) OF '상품_X1' (INDEX) (Cost=1 Card=1)
7  4           INDEX (RANGE SCAN) OF '계약_X2' (INDEX) (Cost=1 Card=1)
8  3         TABLE ACCESS (BY INDEX ROWID) OF '계약' (TABLE) (Cost=1 ... )
```

상품 테이블, 계약 테이블, [상품번호, 계약일자]로 구성된 계약_X2 인덱스가 있을 때

- 위 쿼리는 상품유형코드 조건절에 해당하는 상품에 대해, 계약일자 조건 기간에 발생한 계약 데이터를 전부 읽음
- 상품 수가 적고, 계약 건수가 많을 수록 더 비효율적임
  - 조인 순서가 상품 $\rightarrow$ 계약
  - 실행 계획 순서 :
    - 상품 테이블에서 상품유형코드 = :pclscd 만족하는 상품 찾아
    - 찾은 상품번호로 계약 인덱스에서 계약일자 조건 맞는 로우 찾아
    - 계약 테이블 액세스 해서 계약구분코드 맞는 로우 찾으면서 NL 조인
    - 결과는 중복 제거를 위해 distinct 처리 (Hash Unique)
  - 옵티마이저 동작 :
    - 상품 테이블에서 만약에 상품 5개 찾아
    - 각 상품마다 계약 인덱스/테이블을 통해 조건에 맞는 계약을 찾아
      - NL 조인이라 5개 상품 * 각 상품당 수천 개 계약 = 수만 번 루프 탐색
    - 결과가 중복될 수 있으므로 최종적으로 distinct 처리
  - 계약이 많을 수록 루프 횟수 증가
  - 계약 테이블을 수만 건 이상 인덱스 탐색 + ROWID 접근

이 쿼리를 아래와 같이 바꾸면, Exists 서브쿼리는 데이터의 존재 여부만 확인하면 돼서 조건절을 만족하는 모든 데이터를 읽진 않음.

```sql
select p.상품번호, p.상품명, p.상품가격, ...
from   상품 p
where  p.상품유형코드 = :pclscd
and    EXISTS (select 'X' from 계약 c
               where  c.상품번호 = p.상품번호
               and    c.계약일자 between :dt1 and :dt2
               and    c.계약구분코드 = :ctpcd)
```

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=2 Card=1 Bytes=80)
1  0   FILTER
2  1     NESTED LOOPS (SEMI) (Cost=2 Card=1 Bytes=80)
3  2       TABLE ACCESS (BY INDEX ROWID) OF '상품' (TABLE) (Cost=1 Card=1 ... )
4  3         INDEX (RANGE SCAN) OF '상품_X1' (INDEX) (Cost=1 Card=1)
5  2       TABLE ACCESS (BY INDEX ROWID) OF '계약' (TABLE) (Cost=1 Card=1 ... )
6  5         INDEX (RANGE SCAN) OF '계약_X2' (INDEX) (Cost=1 Card=1)
```

- p.상품유형코드 = :pclscd 에 해당하는 상품에 대해, 계약일자 안에 발생한 계약 중, c.계약구분코드 = :ctpcd 를 만족하는 데이터가 한 건이라도 존재하는지만 확인.
- distinct 연산자를 사용하지 않았으므로 상품 테이블에 대한 부분범위 처리도 가능

<br />

Distinct, Minus 연산자를 사용한 쿼리는 대부분 Exists로 변환 가능.

```sql
-- 튜닝 전
SELECT ST.상황접수번호, ST.관제일련번호, ST.상황코드, ST.관제일시
FROM 관제진행상황 ST
WHERE 상황코드 = '0001' -- 신고접수
AND 관제일시 BETWEEN :V_TIMEFROM || '000000' AND :V_TIMETO || '235959'
MINUS
SELECT ST.상황접수번호, ST.관제일련번호, ST.상황코드, ST.관제일시
FROM 관제진행상황 ST, 구조활동 RPT
WHERE 상황코드 = '0001'
AND 관제일시 BETWEEN :V_TIMEFROM || '000000' AND :V_TIMETO || '235959'
AND RPT.출동센터ID = :V_CNTR_ID
AND ST.상황접수번호 = RPT.상황접수번호
ORDER BY 상황접수번호, 관제일시

-- 튜닝 후
SELECT ST.상황접수번호, ST.관제일련번호, ST.상황코드, ST.관제일시
FROM 관제진행상황 ST
WHERE 상황코드 = '0001' -- 신고접수
AND 관제일시 BETWEEN :V_TIMEFROM || '000000' AND :V_TIMETO || '235959'
AND NOT EXISTS (SELECT 'X' FROM 구조활동 
                WHERE 출동센터ID = :V_CNTR_ID
                AND 상황접수번호 = ST.상황접수번호)
ORDER BY ST.상황접수번호, ST.관제일시
```

- MINUS를 NOT EXISTS로 변경했음

---



### 조인 방식 변경


```sql
select c.계약번호, c.상품코드, p.상품명, p.상품구분코드, c.계약일시, c.계약금액
from   계약 c, 상품 p
where  c.지점ID = :brch_id
and    p.상품코드 = c.상품코드
order by c.계약일시 desc
```


```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS
1  0   SORT (ORDER BY)
2  1     HASH JOIN
3  2       TABLE ACCESS (FULL) OF '상품' (TABLE)
4  2       TABLE ACCESS (BY INDEX ROWID) OF '계약' (TABLE)
5  4         INDEX (RANGE SCAN) OF '계약_X01' (INDEX)
```

- 계약_X1 인덱스가 [지점ID, 계약일시] 순이면 소트 연산을 생략할 수 있지만, 해시 조인이라서 Sort Order By가 일어남

```sql
select /*+ leading(c) use_nl(p) */
       c.계약번호, c.상품코드, p.상품명, p.상품구분코드, c.계약일시, c.계약금액
from   계약 c, 상품 p
where  c.지점ID = :brch_id
and    p.상품코드 = c.상품코드
order by c.계약일시 desc
```

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS
1  0   NESTED LOOPS
2  1     NESTED LOOPS
3  2       TABLE ACCESS (BY INDEX ROWID) OF '계약' (TABLE)
4  3         INDEX (RANGE SCAN DESCENDING) OF '계약_X01' (INDEX)
5  3         INDEX (UNIQUE SCAN) OF '상품_PK' (INDEX (UNIQUE))
6  2       TABLE ACCESS (BY INDEX ROWID) OF '상품' (TABLE)
```

- 계약 테이블 기준으로 상품 테이블과 NL 조인하도록 하면 소트 연산 생략 가능
- 지점ID 조건을 만족하는 데이터가 많고, 부분범위 처리가 가능한 상황에서는 성능 많이 향상

정렬 기준이 조인 키칼럼이면 소트 머지 조인도 Sort Order By 생략 가능

---



## 인덱스를 이용한 소트 연산 생략

인덱스는 이미 정렬되어 있음. Order by, Group by 생략 가능.

Top N  쿼리 특성을 결합하면 온라인 트랜잭션 처리 시스템에서 대량 데이터 조회할 때 빠른 응답 가능.

### Sort Order By 생략

```sql
select 거래일시, 체결건수, 체결수량, 거래대금
from 종목거래
where 종목코드 = 'KR123456'
order by 거래일시
```

- 인덱스를 [종목코드]로 구성하면 소트 연산 생략 불가
  - 종목코드 = 'KR123456' 조건 만족하는 레코드 인덱스에서 전부 읽고, 다 테이블 랜덤 액세스
  - 모든 데이터 읽고 나서 거래일시 순으로 정렬 끝나야 출력 시작 $\rightarrow$ 느려
- 인덱스를 [종목코드, 거래일시] 순으로 구성하면 소트 연산 생략 가능

```
| Id | Operation                  | Name       | Rows  | Bytes | Cost (%CPU)|
|----|----------------------------|------------|-------|-------|------------|
| 0  | SELECT STATEMENT           |            | 40000 | 3515K | 2041  (1)  |
| 1  | SORT ORDER BY              |            | 40000 | 3515K | 2041  (1)  |
| 2  | TABLE ACCESS BY INDEX ROWID| 종목        | 40000 | 3515K | 1210  (1)  |
|* 3 | INDEX RANGE SCAN           | 종목거래_N1  | 40000 |       | 96    (2)  |

Predicate Information (identified by operation id):
---------------------------------------------------
2 - access("종목코드"='KR123456')
```

```
| Id | Operation                  | Name       | Rows  | Bytes | Cost (%CPU)|
|----|----------------------------|------------|-------|-------|------------|
| 0  | SELECT STATEMENT           |            | 40000 | 3515K | 1372  (1)  |
| 1  | TABLE ACCESS BY INDEX ROWID| 종목        | 40000 | 3515K | 1372  (1)  |
|* 2 | INDEX RANGE SCAN           | 종목거래_PK  | 40000 |       | 258   (1)  |

Predicate Information (identified by operation id):
---------------------------------------------------
2 - access("종목코드"='KR123456')
```

- 종목코드 = 'KR123456' 조건을 만족하는 전체 레코드 읽지 않고도 바로 결과집합 출력을 시작할 수 있음
- 부분범위 처리 가능한 상태가 되었음

<br />

요즘은 많은 DB 어플리케이션은 3-Tier 환경에서 작동함.

부분범위 처리 : 쿼리 수행 결과 중 앞쪽 일부를 우선 전송하고, 클라이언트가 추가 전송을 요청하면 남은 데이터를 또 조금 전송

2-Tier 환경에서는 부분범위 처리를 활용한 튜닝을 많이 했음.

요즘은 클라이언트과 DB 사이에 WAS, AP 등이 존재하는 3-Tier 아키텍처가 많음.

​	서버 리소스를 수많은 클라이언트가 공유하는 구조이므로, 클라이언트가 특정 DB 컨넥션을 독점할 수 없음.

​	단위 작업을 마치면 DB 컨넥션을 바로 컨넥션 풀에 반환해야 함.

​	반환하기 전에 쿼리 조회 결과를 클라이언트에게 모두 전송하고 나서 커서를 닫아야 함.

따라서 쿼리 결과 집합을 조금씩 나눠서 전송하는 방식을 사용할 수 없음.

부분범위 처리는 (1) 결과집합 출력을 바로 시작할 수 있는지 (2) 앞쪽 일부만 출력하고 멈출 수 있는지 가 핵심인데, 3-Tier 환경에서 의미가 없다고 생각될 수 있음.

하지만 부분범위 처리는 3-Tier 환경에서도 유효함  $\because$ Top N 쿼리

---



### Top N 쿼리

Top N 쿼리 : 전체 결과집합 중 상위 N개 레코드만 반환

```sql
select * from (
    select 거래일시, 체결건수, 체결수량, 거래대금
    from   종목거래
    where  종목코드 = 'KR123456'
    and    거래일시 >= '20180304'
    order by 거래일시
)
where rownum <= 10
```

- 인라인 뷰로 정의한 집합을 모두 읽어 거래일시 순으로 정렬한 중간집합을 만들고, 거기서 상위 10개 레코드를 취하는 형태
- 소트를 생략할 수 있도록 인덱스를 구성해 줘도 중간집합을 만들어야 하므로, 부분범위 처리는 불가능해 보임.
- [종목코드, 거래일시] 순으로 구성된 인덱스를 이용하면 옵티마이저가 소트 생략

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS
1  0   COUNT (STOPKEY)
2  1     VIEW
3  2       TABLE ACCESS (BY INDEX ROWID) OF '종목거래' (TABLE)
4  3         INDEX (RANGE SCAN) OF '종목거래_PK' (INDEX (UNIQUE))
```

- Sort Order By 대신 COUNT (STOPKEY) 가 있음
- ROWNUM으로 지정한 건수만큼 결과 레코드를 얻으면 거기서 바로 멈춤

#### 페이징

3-Tier 환경에서는 대량의 결과집합을 조회할 때 페이징 처리 기법 사용.

일반적으로 아래와 같은 패턴으로 사용.

```sql
select *
from (
    select rownum no, a.*
    from
    (
        /* SQL Body */
    ) a
    where rownum <= (:page * 10)
)
where no >= (:page-1)*10 + 1
```

3-Tier 환경에서 부분범위를 사용하려면

1. 부분범위 처리 가능하도록 SQL 작성
   - 인덱스 사용 가능하도록 조건절 작성
   - 조인은 NL 위주로 처리
   - Order By 절이 있어도 소트 연산을 생략할 수 있도록 인덱스 구성
2. 작성한 SQL문을 페이징 처리용 표준 패턴 SQL Body 부분에 작성

```sql
select *
from (
    select rownum no, a.*
    from
    (
        select 거래일시, 체결건수, 체결수량, 거래대금
        from   종목거래
        where  종목코드 = 'KR123456'
        and    거래일시 >= '20180304'
        order by 거래일시
    ) a
    where rownum <= (:page * 10)
)
where no >= (:page-1)*10 + 1
```

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=16 Card=756 Bytes=126K)
1  0     VIEW (Cost=16 Card=756 Bytes=126K)
2  1       COUNT (STOPKEY)       --> NO SORT + STOPKEY
3  2         VIEW (Cost=16 Card=756 Bytes=117K)
4  3           TABLE ACCESS (BY INDEX ROWID) OF '종목거래' (TABLE) (Cost=16 ...)
5  4             INDEX (RANGE SCAN) OF '종목거래_PK' (INDEX) (Cost=4 Card=303)
```

- 소트 연산이 없고 COUNT (STOPKEY) 가 있음

#### 페이징 처리 ANTI 패턴

```sql
select *
from (
    select rownum no, a.*
    from
    (
        select 거래일시, 체결건수, 체결수량, 거래대금
        from   종목거래
        where  종목코드 = 'KR123456'
        and    거래일시 >= '20180304'
        order by 거래일시
    ) a
)
where no between (:page-1)*10 + 1 and (:page * 10)
```

- order by 아래쪽 ROWNUM을 제거하고 하나로 합쳐서 간결하게 표현했음
- 그런데 이렇게 사용하면 실행계획이 바뀜

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=16 Card=756 Bytes=126K)
1  0     FILTER
2  1       VIEW (Cost=16 Card=756 Bytes=126K)
3  2         COUNT       --> NO SORT + NO STOP
4  3           VIEW (Cost=16 Card=756 Bytes=117K)
5  4             TABLE ACCESS (BY INDEX ROWID) OF '종목거래' (TABLE) (Cost=16 ...)
6  5               INDEX (RANGE SCAN) OF '종목거래_PK' (INDEX) (Cost=4 Card=303)
```

- COUNT 옆에 (STOPKEY) 가 없어졌음
- 소트 생략 가능하도록 인덱스를 구성했으므로 소트 생략은 가능하지만, STOPKEY가 작동하지 않아 전체 범위 처리
- 마지막에 필터

```sql
    ) a
    where rownum <= (:page * 10) -- 먼저 LIMIT
)
where no >= (:page-1)*10 + 1 -- 이후 OFFSET
```

- 원래는 옵티마이저는 rownum <= (:page * 10) 을 먼저 걸고
- STOPKEY 힌트를 써서, 그 숫자만큼만 가져오고 그 이후는 읽지 않았음

```sql
where no between (:page-1)*10 + 1 and (:page * 10)
```

- Oracle에서 rownum은 레코드가 반환될 때 부여됨
- rownum 자체가 SELECT 이후에 부여되는 가상 칼럼이라, rownum을 계산하기 위해서는 모든 결과를 끝까지 읽어야 함.
- rownum between _ and _ 는 사실상 OFFSET + LIMIT 이 아니라 전부 다 읽은 다음에 필터링 하는 구조

---



### 최소/최대값 구하기

```sql
select MAX(SAL) from EMP;	
```

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=3 Card=1 Bytes=4)
1   0    SORT (AGGREGATE) (Card=1 Bytes=4)
2   1      TABLE ACCESS (FULL) OF 'EMP' (TABLE) (Cost=3 Card=14 Bytes=56)
```

- MIN, MAX 구할 때 Sort Aggregate 나타남
- Sort Aggregate를 위해 전체 데이터를 정렬하진 않지만, 전체 데이터를 읽으면서 값을 비교

#### 인덱스를 통한 최소/최대값 구하기

```
Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=1 Card=1 Bytes=3)
1   0    SORT (AGGREGATE) (Card=1 Bytes=3)
2   1      INDEX (FULL SCAN (MIN/MAX)) OF 'EMP_X1' (INDEX) (Cost=1 Card=1)
```

- 인덱스를 사용하면 전체 데이터를 읽지 않고도 최소/최대값을 쉽게 찾을 수 있음.
  - 인덱스 맨 왼쪽으로 내려가서 첫 번째 읽는 값이 최소값, 오른쪽으로 내려가서 첫 번째 읽는 값이 최대값.
- 인덱스를 이용해 최소/최대값을 찾으려면 조건절 칼럼과 MIN/MAX 함수 인자 칼럼이 모두 인덱스에 포함되어 있어야 함.



---



#### Top N 쿼리를 통한 최소/최대값 구하기

Top N 쿼리에 소트 연산을 생략할 수 있도록 인덱스를 구성했을 때 Top N Stopkey 알고리즘은 성능 향상을 가져올 수 있음.

Top N 쿼리를 통해서도 최소/최대값 구하기 쉬움.

ROWNUM <= 1 조건을 이용해 Top 1 레코드를 찾을 수 있음

```sql
CREATE INDEX EMP_X1 ON EMP(DEPTNO, SAL);

SELECT *
FROM (
    SELECT SAL
    FROM EMP
    WHERE DEPTNO = 30
    AND MGR = 7698
    ORDER BY SAL DESC
)
WHERE ROWNUM <= 1;
```

```
Execution Plan
---------------------------------------------------------
0    SELECT STATEMENT Optimizer=ALL_ROWS (Cost=2 Card=1 Bytes=13)
1    0  COUNT (STOPKEY)
2    1    VIEW (Cost=2 Card=1 Bytes=13)
3    2      TABLE ACCESS (BY INDEX ROWID) OF 'EMP' (TABLE) (Cost=2 Card=1 ...)
4    3        INDEX (RANGE SCAN DESCENDING) OF 'EMP_X1' (INDEX) (Cost=1 Card=5)
```

인덱스는 조건절 칼럼과 MIN/MAX 함수 인자 칼럼이 모두 인덱스에 포함되어 있어야 함.

Top N 쿼리에 작동하는 Top N Stopkey 알고리즘은 모든 칼럼이 인덱스에 포함되어 있지 않아도 잘 작동함.

- 위 인덱스에서 MGR 칼럼은 인덱스에 없음.

- 하지만 DEPTNO = 30 인 모든 레코드를 읽지 않음.

<img src="/assets/images/2025-05-21/Picture7.png" alt="Picture7"/>

- DEPTNO = 30 을 만족하는 가장 오른쪽에서부터 역순으로 스캔하면서 테이블을 액세스하다가 MGR = 7698 조건을 만족하는 레코드를 하나 만나면 멈춤
- 인라인 뷰를 사용하므로 쿼리가 약간 더 복잡하긴 하지만 성능 측면에서는 MIN/MAX 쿼리보다 나음

---



### 이력 조회

일반적으로 테이블은 각 칼럼의 현재 값만 저장.

이전 버전으로 값이 어떻게 변경되어 왔는지 이력을 조회해야 한다면 이력 테이블을 따로 관리해야 함.

<img src="/assets/images/2025-05-21/Picture8.png" alt="Picture7" style="zoom:22%;"/>

이력 테이블에는 보통 현재 테이터도 저장.

장비 테이블에 있는 최종 변경 일자는 칼럼 중 하나라도 변경되면 업데이트 되기 때문에, 특정 칼럼 값이 현재로 바뀐 날짜를 알고 싶으면 이력 테이블에서 찾아야 함.



<br />

이력 데이터를 조회할 때 First Row Stopkey나 [Top N Stopkey](#top-n-쿼리를-통한-최소최대값-구하기) 알고리즘이 작동할 수 있게 인덱스를 설계하고 SQL을 구현해야 한다.


```sql
SELECT 장비번호, 장비명, 상태코드
,(SELECT MAX(변경일자)
  FROM   상태변경이력
  WHERE  장비번호 = P.장비번호) 최종변경일자
FROM   장비 P
WHERE  장비구분코드 = 'A001'
```

- 장비구분코드 = 'A001'인 장비 목록을 조회
- 상태변경이력에서 상태코드가 현재 값으로 변경된 날자 조회

```
| Id | Operation                    | Name         | Starts | A-Rows | Buffers |
|----|------------------------------|--------------|--------|--------|---------|
| 0  | SELECT STATEMENT             |              | 1      | 10     | 4       |
| 1  | SORT AGGREGATE               |              | 10     | 10     | 22      |
| 2  | FIRST ROW                    |              | 10     | 10     | 22      |
| 3  | INDEX RANGE SCAN (MIN/MAX)   | 상태변경이력_PK | 10     | 10     | 22      |
| 4  | TABLE ACCESS BY INDEX ROWID  | 장비          | 1      | 10     | 4       |
| 5  | INDEX RANGE SCAN             | 장비_N1       | 1      | 10     | 2       |
```

- id 2에서 First Row Stopkey
- 상태변경이력_PK 인덱스가 [장비번호, 변경일자, 변경순번] 순으로 구성

<br />

```sql
SELECT 장비번호, 장비명, 상태코드
, SUBSTR(최종이력, 1, 8) 최종변경일자
, TO_NUMBER(SUBSTR(최종이력, 9, 4)) 최종변경순번
FROM (
  SELECT 장비번호, 장비명, 상태코드
  ,(SELECT MAX(H.변경일자 || LPAD(H.변경순번, 4))
    FROM 상태변경이력 H
    WHERE 장비번호 = P.장비번호) 최종이력
  FROM 장비 P
  WHERE 장비구분코드 = 'A001'
)
```

- 최종 변경순번까지 이력 테이블에서 읽음

```
| Id | Operation                     | Name           | Starts | A-Rows | Buffers |
|----|------------------------------ |----------------|--------|--------|---------|
| 0  | SELECT STATEMENT              |                | 1      | 10     | 4       |
| 1  | SORT AGGREGATE                |                | 10     | 10     | 6380    |
| 2  | INDEX RANGE SCAN              | 상태변경이력_PK   | 10     | 1825K  | 6380    |
| 3  | TABLE ACCESS BY INDEX ROWID   | 장비            | 1      | 10     | 4       |
| 4  | INDEX RANGE SCAN              | 장비_N1         | 1      | 10     | 2       |
```

- 인덱스 칼럼을 가공 $\rightarrow$ First Row Stopkey 알고리즘 작동하지 않음



```sql
SELECT 장비번호, 장비명, 상태코드
,(SELECT MAX(H.변경일자)
  FROM 상태변경이력 H
  WHERE 장비번호 = P.장비번호) 최종변경일자
,(SELECT MAX(H.변경순번)
  FROM 상태변경이력 H
  WHERE 장비번호 = P.장비번호
  AND   변경일자 = (SELECT MAX(H.변경일자)
                   FROM 상태변경이력 H
                   WHERE 장비번호 = P.장비번호)) 최종변경순번
FROM 장비 P
WHERE 장비구분코드 = 'A001'
```

- 쿼리를 이렇게 바꾸면 상태변경이력을 세 번 조회하지만 First Row Stopkey는 작동

```
| Id | Operation                     | Name       | Starts | A-Rows | Buffers |
|----|-------------------------------|------------|--------|--------|---------|
| 0  | SELECT STATEMENT              |            | 1      | 10     | 4       |
| 1  | SORT AGGREGATE                |            | 10     | 10     | 22      |
| 2  | FIRST ROW                     |            | 10     | 10     | 22      |
| 3  | INDEX RANGE SCAN (MIN/MAX)    | 상태변경이력  | 10     | 10     | 22      |
| 4  | SORT AGGREGATE                |            | 10     | 10     | 47      |
| 5  | INDEX RANGE SCAN              | 상태변경이력  | 10     | 1000   | 47      |
| 6  | SORT AGGREGATE                |            | 10     | 10     | 22      |
| 7  | FIRST ROW                     |            | 10     | 10     | 22      |
| 8  | INDEX RANGE SCAN (MIN/MAX)    | 상태변경이력  | 10     | 10     | 22      |
| 9  | TABLE ACCESS BY INDEX ROWID   | 장비        | 1      | 10     | 4       |
| 10 | INDEX RANGE SCAN              | 장비_N1     | 1      | 10     | 2       |
```

하지만 테이블에서 읽어야 할 칼럼이 많으면 오히려 더 복잡해짐.



#### INDEX_DESC 힌트 활용

인덱스를 역순으로 읽고, 첫 번째 레코드에서 바로 멈추도록 할 수 있음.

```sql
SELECT 장비번호, 장비명
, SUBSTR(최종이력, 1, 8) 최종변경일자
, TO_NUMBER(SUBSTR(최종이력, 9, 4)) 최종변경순번
, SUBSTR(최종이력, 13) 최종상태코드
FROM (
  SELECT 장비번호, 장비명
  ,(SELECT /*+ INDEX_DESC(X 상태변경이력_PK) */
  					변경일자 || LPAD(변경순번, 4) || 상태코드
  	FROM 상태변경이력 X
  	WHERE 장비번호 = P.장비번호
  	AND ROWNUM <= 1) 최종이력
	FROM 장비 P
	WHERE 장비구분코드 = 'A001'
)
```

```
| Id | Operation                      | Name           | Starts | A-Rows | Buffers |
|----|--------------------------------|----------------|--------|--------|---------|
| 0  | SELECT STATEMENT               |                | 1      | 10     | 4       |
| 1  | COUNT STOPKEY                  |                | 10     | 10     | 41      |
| 2  | TABLE ACCESS BY INDEX ROWID    | 상태변경이력      | 10     | 10     | 41      |
| 3  | INDEX RANGE SCAN DESCENDING    | 상태변경이력      | 10     | 10     | 30      |
| 4  | TABLE ACCESS BY INDEX ROWID    | 장비            | 1      | 10     | 4       |
| 5  | INDEX RANGE SCAN               | 장비_N1         | 1      | 10     | 2       |
```

- 인덱스 구성이 완벽해야만 쿼리가 잘 작동함
- 인덱스 구성이 바뀌면 언제든 결과집합에 문제가 생길 수 있음



#### Predicate Pushing

```sql
SELECT 장비번호, 장비명
, SUBSTR(최종이력, 1, 8) 최종변경일자
, TO_NUMBER(SUBSTR(최종이력, 9, 4)) 최종변경순번
, SUBSTR(최종이력, 13) 최종상태코드
FROM (
  SELECT 장비번호, 장비명
  ,(SELECT 변경일자 || LPAD(변경순번, 4) || 상태코드
    FROM (SELECT 장비번호, 변경일자, 변경순번, 상태코드
          FROM 상태변경이력 -- 여기로 파고 들어감
          ORDER BY 변경일자 DESC, 변경순번 DESC)
    WHERE 장비번호 = P.장비번호  -- 이 조건이 
    AND ROWNUM <= 1) 최종이력
  FROM 장비 P
  WHERE 장비구분코드 = 'A001'
)
```

```
| Id | Operation                      | Name        | Starts | A-Rows | Buffers |
|----|--------------------------------|-------------|--------|--------|---------|
| 0  | SELECT STATEMENT               |             | 1      | 10     | 4       |
| 1  | COUNT STOPKEY                  |             | 10     | 10     | 40      |
| 2  | VIEW                           |             | 10     | 10     | 40      |
| 3  | TABLE ACCESS BY INDEX ROWID    | 상태변경이력   | 10     | 10     | 40      |
| 4  | INDEX RANGE SCAN DESCENDING    | 상태변경이력   | 10     | 10     | 30      |
| 5  | TABLE ACCESS BY INDEX ROWID    | 장비         | 1      | 10     | 4       |
| 6  | INDEX RANGE SCAN               | 장비_N1      | 1      | 10     | 2       |
```

- 인라인 뷰로 정의한 집합(상태변경이력 order by)을 우선 만들고 나서 장비번호와 ROWNUM 조건을 필터링하는 것처럼 보임
- 실제 수행해 보면 장비범호 = P.장비번호 조건절이 인라인 뷰 안쪽으로 파고 들어감
  - predicate pushing이 적용됨
- 이 방식을 이용하면 인덱스가 바뀌었을 때, Top N Stopkey 가 작동하지 않아 성능이 느려질 수는 있지만, 쿼리 결과집합은 보장됨

```sql
SELECT 장비번호, 장비명
, SUBSTR(최종이력, 1, 8) 최종변경일자
, TO_NUMBER(SUBSTR(최종이력, 9, 4)) 최종변경순번
, SUBSTR(최종이력, 13) 최종상태코드
FROM (
    SELECT 장비번호, 장비명
    ,(SELECT 변경일자 || LPAD(변경순번, 4) || 상태코드
      FROM (SELECT 변경일자, 변경순번, 상태코드
            FROM 상태변경이력
            WHERE 장비번호 = P.장비번호 -- 11g까지는 ORA-00904 파싱 에러 떴었음
            ORDER BY 변경일자 DESC, 변경순번 DESC)
      WHERE ROWNUM <= 1) 최종이력
    FROM 장비 P
    WHERE 장비구분코드 = 'A001'
)
```

- 12c 부터는 위 쿼리도 sql 파싱 오류 없이 Top N Stopkey가 작동



---



### Sort Group By 생략

그룹핑 할 때도 인덱스를 활용할 수 있음.

```sql
select region, avg(age), count(*)
from customer
group by region
```

```
| Id | Operation                   | Name        | Rows  | Bytes | Cost (%CPU)|
|----|-----------------------------|-------------|-------|-------|------------|
|  0 | SELECT STATEMENT            |             | 25    | 725   | 30142  (1) |
|  1 | SORT GROUP BY NOSORT        |             | 25    | 725   | 30142  (1) |
|  2 | TABLE ACCESS BY INDEX ROWID | CUSTOMER    | 1000K | 27M   | 30142  (1) |
|  3 |  INDEX FULL SCAN            | CUSTOMER_X01| 1000K |       | 2337   (2) |
```

- region이 선두 칼럼인 인덱스를 이용하면 Sort Group By 연산 생략 가능
- 실행계획에 NOSORT 라고 명시되어 있음

<img src="/assets/images/2025-05-21/Picture9.png" alt="Picture9" style="zoom:22%;" />

운반단위 Array Size = 3 일 때

1. 인덱스에서 'A' 구간을 스캔하면서 테이블을 액세스하다가, 'B'를 만나는 순간, 그때까지 집계한 값을 운반단위에 저장
2. 'B' 구간 스캔하다가 'C'를 만나는 순간, 그때까지 집계한 값을 운반단위에 저장
3. 'C' 구간 스캔하다가 'D'를 만나는 순간, 그때까지 집계한 값을 운반단위에 저장. Array Size = 3 아므로 지금까지 읽은 A, B, C에 대한 집계결과를 클라이언트에게 전송하고, 다음 fetch call 에 대해서 이어서 수행

인덱스를 이용해 NoSort 방식으로 Group By를 처리하면 부분범위 처리가 가능해짐.


---

## Sort Area를 적게 사용하도록 SQL 작성

소트 연산이 불가피하다면 메모리 내에서 처리를 완료할 수 있도록 노력해야 함.

### 소트 데이터 줄이기

특정 기간에 발생한 주문상품 목록을 파일로 내리고자 할 때

```sql
-- [1번]
select lpad(상품번호, 30) || lpad(상품명, 30) || lpad(고객ID, 10)
       || lpad(고객명, 20) || to_char(주문일시, 'yyyymmdd hh24:mi:ss')
from   주문상품
where  주문일시 between :start and :end
order by 상품번호

-- [2번]
select lpad(상품번호, 30) || lpad(상품명, 30) || lpad(고객ID, 10)
       || lpad(고객명, 20) || to_char(주문일시, 'yyyymmdd hh24:mi:ss')
from (
    select 상품번호, 상품명, 고객ID, 고객명, 주문일시
    from   주문상품
    where  주문일시 between :start and :end
    order by 상품번호
)
```

- 2번이 Sort Area를 훨씬 적게 사용
- 1번은 레코드당 107(=30+30+10+20+17) 바이트로 가공한 결과집합을 Sort Area에 저장
- 2번은 가공하지 않은 상태로 정렬을 완료하고 나서 최종 출력할 때 가공

```sql
-- [1번]
SELECT *
FROM 예수금원장
ORDER BY 총예수금 DESC

Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=184K Card=2M Bytes=716M)
1  0   SORT (ORDER BY) (Cost=184K Card=2M Bytes=716M)
2  1     TABLE ACCESS (FULL) OF '예수금원장' (TABLE) (Cost=25K Card=2M Bytes=716M)

-- [2번]
SELECT 계좌번호, 총예수금
FROM   예수금원장
ORDER BY 총예수금 DESC

Execution Plan
---------------------------------------------------------------
0      SELECT STATEMENT Optimizer=ALL_ROWS (Cost=31K Card=2M Bytes=17M)
1  0   SORT (ORDER BY) (Cost=31K Card=2M Bytes=17M)
2  1     TABLE ACCESS (FULL) OF '예수금원장' (TABLE) (Cost=24K Card=2M Bytes=17M)
```

- 2번이 Sort Area 더 적게 사용
- 1번은 모든 칼럼을 Sort Area에 저장 (716MB)
- 2번은 계좌번호과 총예수금만 저장 (17MB)
- 두 쿼리 모두 table full scan 해서 읽은 데이터량은 똑같지만, 소트한 데이터량이 달라서 성능도 다름.

---

### Top N 쿼리의 소트 부하 경감 원리

Top N Stopkey는 인덱스로 소트 연산을 생략할 수 있을 때 작용했음.

인덱스가 없다면?

1000 개 레코드 중에서 특정 칼럼 값이 가장 큰 10개 레코드를 알고 싶을 때

위에서 본 Top N Stopkey 알고리즘 : 정렬된 데이터에서 가장 위쪽에 있는 10개 레코드를 선택

Top N 소트 알고리즘 : 

1. 1000개 중 10개를 잡아서 정렬
2. 다음 990개 레코드를 하나씩 읽으면서 현재 top 10과 비교하고 업데이트

<br />

```sql
select *
from (
    select rownum no, a.*
    from
    (
        select 거래일시, 체결건수, 체결수량, 거래대금
        from   종목거래
        where  종목코드 = 'KR123456'
        and    거래일시 >= '20180304'
        order by 거래일시
    ) a
    where rownum <= (:page * 10)
)
where no >= (:page-1)*10 + 1
```

```
Call     Count CPU Time Elapsed Time  Disk   Query  Current  Rows
------ ------ -------- ------------ ------ ------- --------- -----
Parse       1    0.000        0.000      0       0         0     0
Execute     1    0.000        0.000      0       0         0     0
Fetch       2    0.078        0.083      0     690         0    10
------ ------ -------- ------------ ------ ------- --------- -----
Total       4    0.078        0.084      0     690         0    10

Rows  Row Source Operation
----- -----------------------------------
    0 STATEMENT
   10  COUNT STOPKEY (cr=690 pr=0 pw=0 time=83318 us)
   10   VIEW (cr=690 pr=0 pw=0 time=83298 us)
   10    SORT ORDER BY STOPKEY (cr=690 pr=0 pw=0 time=83264 us)
49857     TABLE ACCESS FULL 종목거래(cr=690 pr=0 pw=0 time=299191 us)
```

- 인덱스로 소트 연산을 생략할 수 없어서 table full scan으로 처리했음
- 종목코드가 선두 칼럼인 인덱스를 사용할 수도 있지만, 바로 뒤 칼럼이 거래일시가 아니면 소트 연산을 생략할 수 없으므로 Sort Order By 오퍼레이션이 수행되었음
- Sort Order By 옆에 STOPKEY가 있음.
  - 소트 연산을 피할 수 없어 Sort Order By 오퍼레이션을 수행하지만 Top N 소트 알고리즘은 작동한다
  - 소트 연산(값 비교) 횟수와 Sort Area 사용량을 최소화해줌
  - page 변수에 1 넣어서 array size = 10이라고 하면, Sort Area에서 필요한 공간은 원소 10개 들어갈 배열 공간만 있으면 됨

1. 처음 읽은 10개 레코드를 거래일시 오름차순으로 정렬해서 배열에 저장
2. 이후 읽는 레코드에 대해서 배열에서 가장 큰 값과 비교해서, 더 작은 값이 나타나면 기존의 맨 끝 값은 버리고 배열 내에서 다시 정렬

```
Statistics
-----------------------
0  recursive calls
0  db block gets
690 consistent gets
0  physical reads
... ...
1  sorts (memory)
0  sorts (disk)
```

- physical reads 와 sorts (disk) 가 0


---

### Top N 쿼리가 아닐 때 발생하는 소트 부하

```sql
select *
from (
    select rownum no, a.*
    from
    (
        select 거래일시, 체결건수, 체결수량, 거래대금
        from   종목거래
        where  종목코드 = 'KR123456'
        and    거래일시 >= '20180304'
        order by 거래일시
    ) a
)
where no between (:page-1)*10 + 1 and (:page * 10)
```

- Order by 아래쪽 ROWNUM 조건 제거

```
Call     Count CPU Time Elapsed Time    Disk   Query    Current   Rows
------- ------ -------- ------------ -------- -------- ---------- ------
Parse       1   0.000       0.000        0       0         0        0
Execute     1   0.000       0.000        0       0         0        0
Fetch       2   0.281       0.858      698     690        14       10
------- ------ -------- ------------ -------- -------- ---------- ------
Total       4   0.281       0.858      698     690        14       10
```

```
Rows Row Source Operation
----- ------------------------------------------------------------
0     STATEMENT
10     VIEW  (cr=690 pr=698 pw=698 time=357962 us)
49857   COUNT  (cr=690 pr=698 pw=698 time=1604327 us)
49857     VIEW  (cr=690 pr=698 pw=698 time=1205452 us)
49857       SORT ORDER BY (cr=690 pr=698 pw=698 time=756723 us)
49857         TABLE ACCESS FULL 중목거래(cr=690 pr=0 pw=0 time=249345 us)
```

- STOPKEY가 없어졌음
- Top N 소트 알고리즘 작동하지 않았음
- 같은 양의 데이터(690 블록)를 읽고 정렬을 수행했는데, 앞에서는 메모리에서 수행하고 여기서는 디스크 이용함

```
Statistics
-----------------------
6  recursive calls
14 db block gets
690 consistent gets
698 physical reads
... ...
0  sorts (memory)
1  sorts (disk)
```

- physical reads 가 698
- sorts (disk) 가 1이므로 정렬 과정에 Temp 테이블스페이스 사용했음

---

### 분석함수에서의 Top N 소트

윈도우 함수 중 rank나 row_number 함수는 max 함수보다 소트 부하가 적음.

Top N 소트 알고리즘이 작동하기 때문.

```sql
select 장비번호, 변경일자, 변경순번, 상태코드, 메모
from (select 장비번호, 변경일자, 변경순번, 상태코드, 메모
      , max(변경순번) over (partition by 장비번호) 최종변경순번
      from 상태변경이력
      where 변경일자 = :upd_dt)
where 변경순번 = 최종변경순번
```

- max 함수를 이용해 모든 장비에 대한 마지막 이력 레코드 찾고 싶음

```
Call     Count CPU Time Elapsed Time    Disk    Query   Current    Rows
------- ------ -------- ------------ --------- -------- --------- -------
Parse       1   0.000      0.000         0        0        0         0
Execute     1   0.000      0.000         0        0        0         0
Fetch       2   2.750      9.175      13456     4536       9        10
------- ------ -------- ------------ --------- -------- --------- -------
Total       4   2.750      9.175      13456     4536       9        10

Rows    Row Source Operation
------- ------------------------------------------------------------
0 STATEMENT
10 VIEW (cr=4536 pr=13456 pw=8960 time=4437847 us)
498570   WINDOW SORT (cr=4536 pr=13456 pw=8960 time=9120662 us)
498570     TABLE ACCESS FULL 상태변경이력 (cr=4536 pr=0 pw=0 time=1994341 us)
```

- Sort Area를 작게 설정한 상태에서 실행
- physical read 13,456
- physical write 8960

```sql
select 장비번호, 변경일자, 변경순번, 상태코드, 메모
from (select 장비번호, 변경일자, 변경순번, 상태코드, 메모
      , rank()over(partitionby장비번호 order by 변경순번 desc) rnum
      from 상태변경이력
      where 변경일자 = :upd_dt)
where rnum = 1
```

- max 대신 rank 사용

```
Call     Count CPU Time Elapsed Time    Disk    Query   Current    Rows
------- ------ -------- ------------ --------- -------- --------- -------
Parse       1   0.000      0.000         0        0        0         0
Execute     1   0.000      0.000         0        0        0         0
Fetch       2   0.969      1.062        40      4536      42        10
------- ------ -------- ------------ --------- -------- --------- -------
Total       4   0.969      1.062        40      4536      42        10

Rows    Row Source Operation
------- ------------------------------------------------------------
0 STATEMENT
10 VIEW (cr=4536 pr=40 pw=40 time=1061996 us)
111   WINDOW SORT PUSHED RANK (cr=4536 pr=40 pw=40 time=1061971 us)
498570     TABLE ACCESS FULL 상태변경이력 (cr=4536 pr=0 pw=0 time=1495760 us)
```

- physical read 40
- physical write 40








<br />

---

출처

친절한 SQL 튜닝




