# OpenCV Ops Guide for Satellite Agent

## 목적
MCP 분석 도구가 반환하는 ops 결과를 어떻게 해석할지 정의한다.

## 지원 연산

### 1) edges
- 설명: Canny edge 기반 경계 검출
- 주요 지표: edge_density
- 해석:
  - 값이 높을수록 경계/윤곽이 많은 장면 가능성
  - 도시/도로/건물 밀집 지역에서 상대적으로 높게 나올 수 있음

### 2) threshold
- 설명: 그레이스케일 임계값 이진화
- 주요 지표: bright_ratio
- 해석:
  - 값이 높을수록 밝은 영역 비중이 큼
  - 구름, 밝은 지붕, 노출된 토양, 반사면이 포함될 수 있음

### 3) morphology
- 설명: 이진 마스크에 대한 morphology open 적용
- 주요 지표: foreground_ratio
- 해석:
  - noise 제거 후 남는 전경 비율
  - 임계값 결과보다 안정적인 면적 추정 참고치로 사용

### 4) cloud_mask_like
- 설명: HSV 범위 기반 cloud-like heuristic
- 주요 지표: cloud_like_ratio
- 해석:
  - 구름 유사 밝은 영역의 비율 추정치
  - 실제 구름 분류 모델이 아니므로 눈/밝은 지면과 혼동 가능

## 주의사항
- 본 ops는 heuristic 기반으로, 정식 원격탐사 분류 모델의 대체재가 아니다.
- 결과는 참고치로 사용하고, 중요한 판단은 추가 검증이 필요하다.
- ROI를 지정하지 않으면 이미지 전체가 분석 대상이다.
