# Satellite Agent FAQ

## Q1. image_uri는 어떤 형식을 지원하나요?
- 로컬 파일 경로와 HTTP/HTTPS URL을 지원한다.
- bytes 직접 전달은 지원하지 않는다.

## Q2. 어떤 질문에서 MCP 분석이 호출되나요?
- 변화, 경계, 구름, 탐지, 분할, 비교 같은 시각 분석 의도가 있고,
  image_uri가 함께 주어졌을 때 호출된다.

## Q3. citations는 무엇인가요?
- RAG 검색으로 찾은 근거 문서 조각이다.
- 각 항목에는 doc_id, chunk_id, snippet, score가 포함된다.

## Q4. analysis 필드는 어떻게 읽나요?
- invoked: MCP가 호출되었는지 여부
- ops: 연산별 요약과 수치 지표
- error: 분석 실패 시 에러 메시지

## Q5. trace는 어떤 용도인가요?
- 내부 디버깅/실험용 필드다.
- 사용된 tool 목록과 처리 지연시간(latency_ms)을 제공한다.

## Q6. 결과가 부정확해 보이면 어떻게 하나요?
- ROI를 좁혀 재분석한다.
- 다른 ops 조합으로 비교한다. (예: edges + cloud_mask_like)
- 가능한 경우 동일 지역의 다른 시점 이미지와 함께 해석한다.
