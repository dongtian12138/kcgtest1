#!/usr/bin/env python3
"""Submit a consultation to one persistent DeepSeek Harness Web session.

Unlike ``dsh --profile headless``, this transport uses the already-running Web
host.  The browser therefore receives the same live session/event frames, and
an observation timeout in this client never cancels the DeepSeek turn.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


DEFAULT_ENDPOINT = "http://127.0.0.1:3080"
DEFAULT_TITLE = "Codex 与 DeepSeek 持续协作（实时）"
_LOCAL_OPENER = build_opener(ProxyHandler({}))


def _rpc(endpoint: str, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    rpc_id = str(uuid4())
    envelope = {
        "type": "client-request",
        "rpcId": rpc_id,
        "method": method,
        "payload": payload,
    }
    request = Request(
        f"{endpoint.rstrip('/')}/api/{method}",
        data=json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Origin": endpoint.rstrip("/"),
        },
        method="POST",
    )
    try:
        # Harness is loopback-only.  Never let HTTP(S)_PROXY divert this local
        # control request to a proxy that cannot see the user's Web process.
        with _LOCAL_OPENER.open(request, timeout=30.0) as reply:
            decoded = json.load(reply)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Harness Web RPC {method} failed: {exc}") from exc
    if decoded.get("rpcId") != rpc_id:
        raise RuntimeError(f"Harness Web RPC {method} returned a mismatched rpcId")
    result = decoded.get("result", {})
    if not result.get("ok"):
        raise RuntimeError(f"Harness Web RPC {method} rejected: {result.get('error')}")
    return result["value"]


def _session_item(endpoint: str, session_id: str) -> dict[str, Any] | None:
    value = _rpc(endpoint, "session.list", {})
    return next(
        (item for item in value["items"] if item["sessionId"] == session_id),
        None,
    )


def _last_seq(item: dict[str, Any] | None) -> int:
    if item is None:
        return -1
    projections = item.get("projections") or {}
    return int(projections.get("asOfSeq", -1))


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_state(path: Path, endpoint: str, session_id: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "kcg_deepseek_harness_live_session_v1",
                "endpoint": endpoint,
                "session_id": session_id,
                "title": title,
                "browser_hint": endpoint,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _latest_assistant_text(endpoint: str, session_id: str, after_seq: int) -> str:
    history = _rpc(
        endpoint,
        "session.history",
        {"sessionId": session_id, "maxMessages": 1},
    )
    candidates: list[tuple[int, str]] = []
    for entry in history["events"]:
        event = entry["event"]
        if event["seq"] <= after_seq or event["type"] != "assistant/message":
            continue
        message = event.get("data", {}).get("message", {})
        text_parts = [
            part.get("text", "")
            for part in message.get("content", [])
            if part.get("type") == "text"
        ]
        candidates.append((event["seq"], "\n\n".join(filter(None, text_parts))))
    if not candidates:
        return ""
    return max(candidates, key=lambda pair: pair[0])[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--session-id")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="observation timeout only; it never cancels the Harness turn",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument(
        "--expose-repository",
        action="store_true",
        help="create a Web session rooted at this repository (user authorization required)",
    )
    args = parser.parse_args()

    problem = args.problem.resolve()
    response = args.response.resolve()
    response.parent.mkdir(parents=True, exist_ok=True)
    state_file = (
        args.state_file.resolve()
        if args.state_file is not None
        else response.parent / ".harness_live_session.json"
    )
    endpoint = args.endpoint.rstrip("/")
    state = _load_state(state_file)
    session_id = args.session_id or state.get("session_id")
    item = _session_item(endpoint, session_id) if session_id else None

    if item is None:
        if not args.expose_repository:
            parser.error(
                "no live session exists; pass --expose-repository only after the user "
                "has authorized remote repository exposure"
            )
        created = _rpc(
            endpoint,
            "session.create",
            {"cwd": str(Path(__file__).resolve().parents[1])},
        )
        session_id = created["sessionId"]
        _rpc(
            endpoint,
            "session.rename",
            {"sessionId": session_id, "title": args.title},
        )
        item = _session_item(endpoint, session_id)

    assert session_id is not None
    _write_state(state_file, endpoint, session_id, args.title)
    before_seq = _last_seq(item)
    started = datetime.now(timezone.utc)
    monotonic_start = time.monotonic()
    prompt = problem.read_text(encoding="utf-8")
    _rpc(
        endpoint,
        "session.prompt",
        {
            "sessionId": session_id,
            "mode": "queue",
            "content": [{"type": "text", "text": prompt}],
            "clientTimeZone": "Etc/UTC",
        },
    )

    print(
        json.dumps(
            {
                "accepted": True,
                "session_id": session_id,
                "title": args.title,
                "browser": endpoint,
                "live_events": [
                    f"ws://{endpoint.split('://', 1)[-1]}/api/events.mux",
                    f"ws://{endpoint.split('://', 1)[-1]}/api/events.host",
                ],
                "note": "请在 Harness 页面打开固定标题；本客户端超时不会停止任务",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    completed = False
    observation_timeout = False
    if not args.no_wait:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            current = _session_item(endpoint, session_id)
            if (
                current is not None
                and not current["running"]
                and _last_seq(current) > before_seq
            ):
                completed = True
                break
            time.sleep(max(0.25, args.poll_interval))
        if not completed:
            observation_timeout = True

    final_text = (
        _latest_assistant_text(endpoint, session_id, before_seq) if completed else ""
    )
    if completed:
        response.write_text(final_text + ("\n" if final_text else ""), encoding="utf-8")
    else:
        response.write_text(
            "该轮已提交到持久 Harness Web 会话，仍在运行或未等待；没有取消任务。\n",
            encoding="utf-8",
        )

    current = _session_item(endpoint, session_id)
    metadata = {
        "schema_version": "kcg_deepseek_consultation_v2",
        "started_at_utc": started.isoformat(),
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "observation_duration_s": time.monotonic() - monotonic_start,
        "interface": "persistent Harness Web RPC + browser WebSocket event stream",
        "endpoint": endpoint,
        "harness_session_id": session_id,
        "harness_session_title": args.title,
        "harness_running_at_observation_end": (
            current.get("running") if current is not None else None
        ),
        "turn_completed": completed,
        "observation_timeout_task_continues": observation_timeout,
        "repository_exposure_authorized": bool(args.expose_repository),
        "workspace_exposure": "full repository; explicitly authorized by user",
        "network_boundary": "deepseek-official remote provider via local Harness host",
        "state_file": str(state_file),
        "browser_hint": endpoint,
        "failure": None if completed or args.no_wait else "observation_timeout_only",
    }
    response.with_suffix(".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return 0 if completed or args.no_wait else 3


if __name__ == "__main__":
    raise SystemExit(main())
