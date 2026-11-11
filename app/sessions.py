"""One simulation per browser session.

Each browser gets its own ``LOBManager``, found through a random session ID
kept in an HTTP-only cookie. Tabs of the same browser share the cookie and so
share a simulation.
"""

from __future__ import annotations

import secrets
import time
from typing import Optional

from starlette.datastructures import MutableHeaders
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.manager import LOBManager
from app.studies import StudyBoard
from app.triggers import TriggerBoard

COOKIE_NAME = "lob_session"


class SessionRegistry:
    """Simulations keyed by session ID.

    IDs are only ever issued here: an unknown or expired ID gets a fresh
    session under a new ID, so a client cannot choose its own. Sessions with
    no open event stream are evicted after ``idle_timeout`` seconds without a
    request, and the least recently seen session is evicted when a new one
    would exceed ``max_sessions``. Evicting a session stops its simulation. Each manager is given a board of
    microstructure studies (app/studies.py) and one of trigger rules on them (app/triggers.py).
    """

    def __init__(self, max_sessions: int = 50, idle_timeout: float = 30 * 60):
        self.max_sessions = max_sessions
        self.idle_timeout = idle_timeout
        self._managers: dict[str, LOBManager] = {}
        self._last_seen: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._managers)

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._managers

    def get(self, session_id: Optional[str]) -> Optional[LOBManager]:
        return self._managers.get(session_id) if session_id else None

    def touch(self, session_id: str) -> None:
        if session_id in self._managers:
            self._last_seen[session_id] = time.monotonic()

    def get_or_create(self, session_id: Optional[str]) -> tuple[str, LOBManager, bool]:
        """Return (session_id, manager, created) for a request's session cookie."""
        now = time.monotonic()
        self._evict_idle(now)
        if session_id in self._managers:
            self._last_seen[session_id] = now
            return session_id, self._managers[session_id], False
        while self._managers and len(self._managers) >= self.max_sessions:
            self._evict(min(self._last_seen, key=self._last_seen.__getitem__))
        session_id = secrets.token_urlsafe(16)
        manager = LOBManager()
        manager.attach_studies(StudyBoard())  # a session's own; a scratch manager of a fit has none
        manager.attach_triggers(TriggerBoard())  # its rules, which read the studies' values
        self._managers[session_id] = manager
        self._last_seen[session_id] = now
        return session_id, manager, True

    def _evict_idle(self, now: float) -> None:
        idle = [sid for sid, seen in self._last_seen.items()
                if now - seen > self.idle_timeout and not self._managers[sid].has_subscribers]
        for sid in idle:
            self._evict(sid)

    def _evict(self, session_id: str) -> None:
        manager = self._managers.pop(session_id)
        self._last_seen.pop(session_id, None)
        manager.shutdown()


class SessionMiddleware:
    """Attach each request to its session, issuing a cookie for new sessions.

    The session ID is stored in ``request.state.session_id``. Paths starting
    with one of ``exempt_prefixes`` (static files) never create sessions.
    Written as plain ASGI so streaming responses pass through untouched.
    """

    def __init__(self, app: ASGIApp, registry: SessionRegistry,
                 exempt_prefixes: tuple[str, ...] = ("/static/", "/favicon.ico")):
        self.app = app
        self.registry = registry
        self.exempt_prefixes = exempt_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"].startswith(self.exempt_prefixes):
            await self.app(scope, receive, send)
            return

        cookie = HTTPConnection(scope).cookies.get(COOKIE_NAME)
        session_id, _, created = self.registry.get_or_create(cookie)
        scope.setdefault("state", {})["session_id"] = session_id
        if not created:
            await self.app(scope, receive, send)
            return

        set_cookie = f"{COOKIE_NAME}={session_id}; Path=/; HttpOnly; SameSite=Lax"

        async def send_with_cookie(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("set-cookie", set_cookie)
            await send(message)

        await self.app(scope, receive, send_with_cookie)
