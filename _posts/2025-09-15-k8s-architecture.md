---
title: "[Cloud] 쿠버네티스 아키텍처 개요"
date: 2025-09-15
categories: ["Cloud"]
tags: ["kubernetes", "cluster", "container"]
use_math: true
---

<img src="/assets/images/2025-09-15/kubernetes-cluster-architecture.svg" alt="kubernetes-cluster-architecture"/>
<span style="color:gray; font-size: 80%">출처 : https://kubernetes.io/docs/concepts/architecture/</span>


쿠버네티스 클러스터(인스턴스 단위) 안에는 control plane과 worker nodes가 존재

## Control Plane

- **Cloud Controller Manager** : 쿠버네티스 클러스터를 클라우드 제공업체의 API에 연결하여 클라우드별 리소스를 관리하고 기본 인프라와의 적절한 통합을 보장
    - 클라우드 플랫폼과 상호작용하는 구성요소와 클러스터 내부에서만 작동하는 구성요소를 분리
    - 클라우드 제공업체별 컨트롤러만 실행 (AWS, GCP, Azure 등에 따라 다름)
    - 온프레미스 환경이나 개인 PC의 학습 환경에서는 필요하지 않음
    - 여러 개의 논리적으로 독립적인 제어 루프를 하나의 바이너리로 결합
    - 성능 향상과 장애 대응을 위한 수평적 확장 가능

- **etcd** : 클러스터 상태에 대한 모든 데이터를 저장하는 키-값 저장소
    - kube-apiserver 만이 etcd 와 직접 통신할 수 있음
    - 이걸 backing store로 사용할 거면 백업 잘 해야 함

- **kube-apiserver** : 쿠버네티스 API를 노출하고, 대부분의 요청을 처리하며, API 요청을 처리하고 검증하여 클러스터와의 상호작용을 관리하는 쿠버네티스의 핵심 구성 요소
    - etcd와 직접 통신하는 유일한 구성요소 (kube-apiserver -> etcd)

- **kube-controller-manager** : 컨트롤러 프로세스들을 실행하는 control plane 구성요소
    - 쿠버네티스 클러스터의 상태를 모니터링하며, 현재 상태가 원하는 상태와 일치하도록 보장하는 프로세스들을 실행
    - 논리적으로는 다양한 컨트롤러들이 있는데, 복잡성을 줄이기 위해 모두 단일 바이너리로 컴파일되어 단일 프로세스로 실행
        - Node controller : 노드 다운 시 이를 감지하고 대응
        - Job controller : job가 생성될 때 필요한 Pods 생성, Pod 상태 확인, 병렬 작업 관리, etc.
        - TTL-after-finished controller : job이 끝나면 해당 job과 pods cleanup
        - EndpointSlice controller : EndpointSlice 객체를 채워서 서비스와 pod 간 연결 제공
        - ServiceAccount controller : 새 네임스페이스를 위한 기본 ServiceAccount 생성
        - etc.

- **kube-scheduler** : Worker Node에 할당되지 않은 새로 생성된 Pod를 식별하고 특정 노드에 할당
    - Pod 스케줄링 결정을 위한 정보 조회


## Worker Nodes 

- **kubelet** : 각 노드에서 pod 내 컨테이너가 정상적으로 실행되고 있는지 확인
    - Orchestrator - **worker node의 control plane**
    - 노드 관리
        - 노드 등록 : 클러스터에 새 노드 추가 시 자신을 등록
        - 노드 상태 보고: CPU, 메모리, 디스크, 네트워크 등 노드 리소스 상태 주기적 전송
        - 노드 heartbeat: 노드가 살아있음을 주기적으로 알림 (기본 10초마다)
        - Node Lease 갱신: 노드 가용성 상태 업데이트
    - Pod 라이프사이클 관리
        - Pod 할당 받기: API Server에서 해당 노드에 스케줄된 새로운 Pod 정보 조회
            - Container Runtime에게 이미지 pull 명령 전달, 이미지 pull 진행상황 모니터링, pull 완료 후 컨테이너 생성 명령
        - Pod 상태 보고: Running, Pending, Failed, Succeeded 등 Pod phase 업데이트
        - 컨테이너 상태 보고: 각 컨테이너의 상태, 재시작 횟수, 종료 이유 등
        - Pod 삭제 확인: Pod 삭제 작업 완료 후 API Server에 확인 신호
    - 실시간 모니터링을 통한 변화 감지, Pod 생성/실행/오류 등의 이벤트를 API Server에 전송, 컨테이너 crash, 이미지 pull 실패 등 알림
    - 리소스 정보 수집
    - 컨트롤 플레인의 kube-apiserver와 통신하여 노드의 원하는 상태를 유지

- **Container Runtime** : 컨테이너 이미지를 가져오고, 컨테이너를 생성 및 관리하며, 쿠버네티스 컨트롤 플레인의 지시에 따라 컨테이너가 적절하고 안전하게 실행되도록 보장
    - kubelet이 관리자라면, Container Runtime은 **실제 작업자**
    - kube-apiserver와는 kubelet을 통해 간접 통신

- **kube-proxy** : 쿠버네티스 클러스터의 각 노드에서 실행되는 네트워크 프록시
    - kube-apiserver를 watch하여 Service와 Endpoint 객체의 변화를 실시간으로 감지
    - Service와 Endpoint 정보를 가져와서 네트워크 규칙 설정, 유지
    - node 및 control plan 내에서 pod와 서비스 간 통신을 가능하게 함




---


## pod 생성 flow

1. 요청 접수 (kubectl → kube-apiserver)

    ```bash
    kubectl apply -f pod.yaml
    ```

    - kubectl이 Pod manifest를 kube-apiserver로 전송
    - kube-apiserver가 요청 검증 (인증, 권한, 리소스 스키마 등)
    - 검증 통과 시 etcd에 Pod 객체 저장 (상태: Pending)

2. 스케줄링 (kube-scheduler)

    - kube-scheduler가 새로운 Pod 생성을 감지 (API Server watch)
    - 적절한 노드 선택:
        - 리소스 요구사항 확인 (CPU, 메모리)
        - 노드 선택기(nodeSelector), 어피니티 규칙 적용
        - Tains/Tolerations 확인

    - 선택된 노드 정보를 kube-apiserver에 업데이트

3. Pod 실행 (kubelet)
    - 해당 노드의 kubelet이 동작:
        - API Server에서 자신의 노드에 할당된 새 Pod 감지
        - Container Runtime에게 이미지 pull 요청
        - 이미지 다운로드 완료 후:
            - 네트워크 설정 (IP 할당)
            - 볼륨 마운트
            - 컨테이너 생성 및 실행

        - Pod 상태를 kube-apiserver에 지속적으로 보고

4. 네트워킹 설정 (kube-proxy)

    - kube-proxy가 Service 연결 감지
    - 새 Pod이 Service의 endpoint에 추가됨을 확인
    - 해당 노드의 iptables/IPVS 규칙 업데이트
    - 트래픽 라우팅 규칙 적용

5. 컨트롤러 동작 (kube-controller-manager)

    - Deployment Controller: ReplicaSet 관리
    - ReplicaSet Controller: Pod 수 유지
    - Endpoint Controller: Service endpoint 업데이트
    - 기타 Controller들: 각각의 리소스 상태 관리

6. 상태 업데이트 과정

    ```
    Pending → ContainerCreating → Running
    ```

    - 각 단계마다 kubelet이 kube-apiserver에 상태 보고
    - etcd에 최신 상태 저장
    - kubectl get pods로 실시간 상태 확인 가능


### 핵심 구성요소 역할 요약:
- kube-apiserver: 중앙 조정자, 모든 통신의 허브
- etcd: 상태 저장소
- kube-scheduler: 노드 배치 결정
- kubelet: 실제 Pod 실행 및 관리
- Container Runtime: 이미지 pull 및 컨테이너 실행
- kube-proxy: 네트워크 규칙 설정
- Controllers: 원하는 상태 유지

