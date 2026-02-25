# cursor_bridge

Cursor CLI를 **[OpenAI 호환 HTTP API](https://docs.openclaw.ai)** 로 노출합니다 (OpenClaw 또는 `/v1/chat/completions` 를 쓰는 클라이언트용).

- **라이선스**: Apache-2.0

**English** | [README.md](README.md)

---

## 개요

```
[OpenClaw 등 클라이언트]  --HTTP (OpenAI 형식)-->  [cursor_bridge :18081]  --하위 프로세스-->  [cursor agent]
```

- FastAPI 앱: `/v1/chat/completions`, `/v1/models`, `/health`
- 요청을 `cursor agent --print --output-format json ...` 로 전달하고, 파싱된 텍스트를 OpenAI 응답 형태로 반환

## 요구 사항

- **Cursor CLI** 가 `PATH` 에 있고 `cursor agent` 가 동작할 것
- **Python 3** 및 의존성: `pip install -r requirements.txt` (FastAPI, uvicorn, pydantic v2)
- 선택: Cursor 인증용 `CURSOR_API_KEY` 를 환경 변수(또는 `.env` / `cursor_bridge.env`)에 설정

## 실행

```bash
# 저장소 루트에서
pip install -r requirements.txt
./run.sh
# => uvicorn cursor_bridge:app --host 127.0.0.1 --port 18081
```

가상환경 사용(의존성 충돌 시 권장):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run.sh   # run.sh 는 .venv 가 있으면 사용
```

비밀값은 `.env` 또는 `cursor_bridge.env` 에 두세요(저장소에 넣지 말 것). `run.sh` 가 로드합니다.

## 확인

```bash
curl -s http://127.0.0.1:18081/health
curl -s http://127.0.0.1:18081/v1/models
```

## OpenClaw

`openclaw.json` 에 커스텀 프로바이더 추가:

- `baseUrl`: `http://127.0.0.1:18081/v1`
- `apiKey`: 예: `dummy` (브릿지는 기본적으로 검증하지 않음)

## 환경 변수 / 옵션

- `CURSOR_BRIDGE_DEBUG=1` — 상세 로그
- `CURSOR_BRIDGE_TIMEOUT` — `cursor agent` 대기 초 (기본 180)
- `CURSOR_BRIDGE_WORKSPACE` — cursor agent 에 전달할 `--workspace` 경로
- 자세한 내용은 `run.sh` 와 소스 참고.

## 문서

- **OpenClaw** 배포·트러블슈팅·422/타임아웃: OpenClaw 워크스페이스의 `docs/cursor_bridge/` 또는 프로젝트 문서 참고 (예: [custom-cursor-setup](https://docs.openclaw.ai) 패턴).

## 트러블슈팅

- **ImportError (pydantic/FastAPI)**: `pip install -r requirements.txt`
- **"Cursor agent is not logged in"**: `.env` 또는 `cursor_bridge.env` 에 `CURSOR_API_KEY` 설정 후 `./run.sh` 로 실행
- API 키 관련: `TROUBLESHOOTING-CURSOR-API-KEY.ko.md` 참고.
