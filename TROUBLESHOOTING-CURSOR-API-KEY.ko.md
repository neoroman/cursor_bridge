# "Cursor agent is not logged in" / CURSOR_API_KEY 오류

**English** | [TROUBLESHOOTING-CURSOR-API-KEY.md](TROUBLESHOOTING-CURSOR-API-KEY.md)

---

## 오류 메시지

```
Cursor agent is not logged in. Run 'cursor agent' in a terminal and complete login,
or set CURSOR_API_KEY in the environment.
```

**유사하게**, "The provided API key is **invalid**" 가 나오면 키는 전달됐지만 값이 잘못된 것입니다.  
→ **workspace/docs/cursor_bridge/cursor-api-key-invalid.md** 참고.

이 메시지는 **cursor_bridge** 자체가 아니라 **cursor agent CLI** 하위 프로세스의 stderr/stdout 에서 감지됩니다.  
즉, cursor_bridge 가 `cursor agent` 를 실행할 때 CLI 가 "login required" 또는 "authentication required" 를 출력하면 위 오류로 변환됩니다.

## CURSOR_API_KEY 를 설정해도 실패하는 이유

`CURSOR_API_KEY` 는 **cursor_bridge 프로세스(uvicorn) 환경**에 있어야 하고,  
cursor_bridge 는 그 환경을 **cursor agent 하위 프로세스**에 그대로 전달합니다.

따라서 다음이 만족되어야 합니다.

1. **항상 `./run.sh` 로 cursor_bridge 를 시작**  
   - `run.sh` 는 `.env`, `cursor_bridge.env`, **`.env/cursor_bridge.env`** 순으로 로드합니다.  
   - 여기에 설정한 `CURSOR_API_KEY` 가 셸 환경에 들어가고, `exec uvicorn ...` 를 통해 uvicorn(cursor_bridge)에 전달됩니다.

2. **run.sh 없이 시작하면 키가 로드되지 않음**  
   - 예: `uvicorn cursor_bridge:app --host 127.0.0.1 --port 18081` 만 실행  
   - 예: systemd / pm2 / OpenClaw 가 run.sh 없이 python/uvicorn 만 실행  
   → `.env` / `.env/cursor_bridge.env` 가 로드되지 않고 **CURSOR_API_KEY 가 비어 있음**,  
   → cursor agent 하위 프로세스가 빈 환경을 물려받아 "not logged in" 발생.

3. **.env 파일 위치와 형식**  
   - 경로: `cursor_bridge/.env/cursor_bridge.env` (`.env` 디렉터리 안의 `cursor_bridge.env` 파일)  
   - 형식: 한 줄 `CURSOR_API_KEY=value` (`=` 앞뒤 공백 없음, 공백이 있으면 값 따옴표)  
   - `run.sh` 는 `set -a` 후 source 하므로 자식 프로세스에 별도 `export` 불필요.

## 확인 방법

1. **cursor_bridge 시작 로그**  
   - 최근 cursor_bridge 시작 시:  
     `cursor_bridge startup: CURSOR_API_KEY in process env=set` 또는 `NOT SET`  
   - `NOT SET` 이면 현재 프로세스에 키가 없음. **run.sh 로 재시작**.

2. **실제로 run.sh 로 시작했는지 확인**  
   - 터미널: `cd /path/to/cursor_bridge && ./run.sh`  
   - systemd/pm2: `ExecStart` 가 `.../cursor_bridge/run.sh` 를 쓰는지 확인.  
     uvicorn 을 직접 실행 중이면 run.sh 로 바꾸거나  
     `Environment=CURSOR_API_KEY=...` 또는 `EnvironmentFile=.../cursor_bridge/.env/cursor_bridge.env` 추가.

3. **Cursor CLI 가 CURSOR_API_KEY 를 쓰는지**  
   - Cursor 문서/버전에 따라 헤드리스에서 `CURSOR_API_KEY` 지원 여부가 다를 수 있음.  
   - 지원하지 않으면 터미널에서 `cursor agent` 를 한 번 실행해 로그인한 뒤, 같은 사용자/머신에서 run.sh 로 cursor_bridge 를 시작해 세션을 공유.

## 요약

| 원인 | 조치 |
|------|------|
| run.sh 없이 cursor_bridge 시작 | **항상 `./run.sh` 로 시작** (또는 시작 시 동일한 환경 주입) |
| .env 경로/파일명 오타 | `cursor_bridge/.env/cursor_bridge.env` 존재 및 이름 확인 |
| .env 형식 오류 | `CURSOR_API_KEY=value` (공백 없음, export 불필요) |
| systemd/pm2 가 uvicorn 만 실행 | ExecStart 를 run.sh 로 변경하거나 CURSOR_API_KEY 를 Environment/EnvironmentFile 로 전달 |

재시작 후에도 오류가 나면 cursor_bridge 로그에서 `CURSOR_API_KEY in process env=` 가 `set` 인지 `NOT SET` 인지 확인하세요.
