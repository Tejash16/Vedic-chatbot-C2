#!/usr/bin/env python3
"""Message bus for the C2 benchmark sessions. Standard library only.

Messages are small text files in bus/messages, carried between laptops by Git.
Each session has its own clone of the repo and a .session file holding its letter.

Commands (run from the clone root):
  python bus/bus.py post --to C --type QUESTION --subject "chunk ids" --text "..."
  python bus/bus.py inbox
  python bus/bus.py wait --type READY --from C --subject B_READY
  python bus/bus.py status "embedding corpus, 40% done"
  python bus/bus.py board
  python bus/bus.py gpu acquire
  python bus/bus.py gpu release
  python bus/bus.py push results/A/results.csv results/A/findings.md -m "A results"
"""
import argparse
import datetime
import os
import random
import string
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MSG_DIR = ROOT / "bus" / "messages"
STATUS_DIR = ROOT / "bus" / "status"
SEEN_FILE = ROOT / ".bus_seen"
LOCAL = ROOT.parent / "local"
GPU_LOCK = LOCAL / "gpu.lock"

SESSIONS = ("A", "B", "C", "D")
TYPES = ("INFO", "QUESTION", "ANSWER", "READY", "BLOCKED", "DONE")
STALE_LOCK_SECONDS = 4 * 3600


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def me():
    f = ROOT / ".session"
    if not f.exists():
        sys.exit("No .session file in the clone root. It must contain one letter: A, B, C or D.")
    s = f.read_text(encoding="utf-8").strip().upper()
    if s not in SESSIONS:
        sys.exit(".session must contain A, B, C or D. Found: %r" % s)
    return s


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError("git %s failed:\n%s\n%s" % (" ".join(args), r.stdout, r.stderr))
    return r


def pull(quiet=True):
    r = git("pull", "--rebase", "--autostash", check=False)
    if r.returncode != 0 and not quiet:
        print("WARNING: git pull failed:\n" + r.stdout + r.stderr, file=sys.stderr)
    return r.returncode == 0


def push_paths(paths, message):
    """Commit only the given paths and push, retrying when the other sessions pushed first."""
    rel = [str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/") for p in paths]
    git("add", "--", *rel)
    git("commit", "-m", message, "--", *rel, check=False)  # "nothing to commit" is fine
    last = None
    for attempt in range(6):
        pull()
        last = git("push", check=False)
        if last.returncode == 0:
            return True
        time.sleep(2 + attempt * 3)
    print("WARNING: push failed after 6 tries. The change is committed locally.\n"
          + (last.stdout + last.stderr if last else ""), file=sys.stderr)
    return False


def parse(path):
    head, body, in_body = {}, [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if in_body:
            body.append(line)
        elif line.strip() == "":
            in_body = True
        elif ":" in line:
            k, v = line.split(":", 1)
            head[k.strip().lower()] = v.strip()
    head["id"] = path.stem
    head["body"] = "\n".join(body).strip()
    return head


def all_messages():
    if not MSG_DIR.exists():
        return []
    return [parse(p) for p in sorted(MSG_DIR.glob("*.md"))]


def show(m):
    print("-" * 70)
    print("id: %s" % m["id"])
    print("from %s to %s | %s | %s" % (m.get("from"), m.get("to"), m.get("type"), m.get("time")))
    if m.get("subject"):
        print("subject: %s" % m["subject"])
    if m.get("re"):
        print("in reply to: %s" % m["re"])
    print()
    print(m["body"])


def load_seen():
    if SEEN_FILE.exists():
        return set(SEEN_FILE.read_text(encoding="utf-8").split())
    return set()


def save_seen(seen):
    SEEN_FILE.write_text("\n".join(sorted(seen)), encoding="utf-8")


def cmd_post(a):
    sender = me()
    to = a.to.upper()
    if to not in SESSIONS + ("ALL",):
        sys.exit("--to must be A, B, C, D or ALL")
    MSG_DIR.mkdir(parents=True, exist_ok=True)
    t = now_utc()
    rand = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    name = "%s_%s-to-%s_%s_%s.md" % (t.strftime("%Y%m%dT%H%M%SZ"), sender, to, a.type, rand)
    path = MSG_DIR / name
    text = a.text
    if a.file:
        text = Path(a.file).read_text(encoding="utf-8")
    lines = ["from: " + sender, "to: " + to, "type: " + a.type,
             "subject: " + (a.subject or ""), "re: " + (a.re or ""),
             "time: " + t.strftime("%Y-%m-%dT%H:%M:%SZ"), "", text or ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok = push_paths([path], "bus: %s to %s %s %s" % (sender, to, a.type, a.subject or ""))
    print("posted %s%s" % (path.stem, "" if ok else " (NOT pushed yet)"))


def cmd_inbox(a):
    who = me()
    pull(quiet=False)
    seen = load_seen()
    msgs = all_messages()
    if a.all:
        picked = msgs[-a.last:]
    else:
        picked = [m for m in msgs if m["id"] not in seen
                  and m.get("from") != who and m.get("to") in (who, "ALL")]
    if not picked:
        print("No new messages for %s." % who)
    for m in picked:
        show(m)
        seen.add(m["id"])
    save_seen(seen)


def matches(m, a, who):
    if m.get("type") != a.type:
        return False
    if a.sender and m.get("from") != a.sender.upper():
        return False
    if a.subject and a.subject.lower() not in (m.get("subject") or "").lower():
        return False
    return m.get("to") in (who, "ALL") or a.any_recipient


def cmd_wait(a):
    who = me()
    deadline = time.time() + a.timeout
    while True:
        pull()
        for m in all_messages():
            if matches(m, a, who):
                show(m)
                seen = load_seen()
                seen.add(m["id"])
                save_seen(seen)
                return
        if time.time() >= deadline:
            print("Not yet. No %s message matched within %d seconds. Run the same wait again."
                  % (a.type, a.timeout))
            sys.exit(2)
        time.sleep(a.interval)


def cmd_status(a):
    who = me()
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATUS_DIR / (who + ".md")
    path.write_text("session %s | %s\n\n%s\n" % (who, now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"), a.text),
                    encoding="utf-8")
    push_paths([path], "status: %s" % who)
    print("status updated")


def cmd_board(a):
    pull(quiet=False)
    print("=== STATUS ===")
    for s in SESSIONS:
        p = STATUS_DIR / (s + ".md")
        print(p.read_text(encoding="utf-8").strip() if p.exists() else "session %s | no status yet" % s)
        print()
    print("=== LAST %d MESSAGES ===" % a.last)
    for m in all_messages()[-a.last:]:
        print("%s | %s to %s | %s | %s" % (m.get("time"), m.get("from"), m.get("to"),
                                           m.get("type"), m.get("subject") or m["body"][:60]))


def cmd_push(a):
    ok = push_paths(a.paths, a.message)
    print("pushed" if ok else "NOT pushed")
    sys.exit(0 if ok else 1)


def read_lock():
    try:
        owner, stamp = GPU_LOCK.read_text(encoding="utf-8").split()
        return owner, float(stamp)
    except Exception:
        return None, 0.0


def cmd_gpu(a):
    who = me()
    if not LOCAL.exists():
        sys.exit("Folder not found: %s" % LOCAL)
    if a.action == "release":
        owner, _ = read_lock()
        if owner in (who, None) or a.force:
            try:
                GPU_LOCK.unlink()
            except FileNotFoundError:
                pass
            print("gpu released")
        else:
            sys.exit("GPU lock is held by session %s, not released." % owner)
        return
    deadline = time.time() + a.timeout
    while True:
        try:
            fd = os.open(str(GPU_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, ("%s %f" % (who, time.time())).encode("utf-8"))
            os.close(fd)
            print("gpu acquired by %s" % who)
            return
        except FileExistsError:
            owner, stamp = read_lock()
            if owner == who:
                print("gpu already held by %s" % who)
                return
            if time.time() - stamp > STALE_LOCK_SECONDS:
                try:
                    GPU_LOCK.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.time() >= deadline:
                print("GPU is busy (held by session %s). Run the same command again." % owner)
                sys.exit(2)
            time.sleep(10)


def main():
    p = argparse.ArgumentParser(description="C2 message bus")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("post")
    s.add_argument("--to", required=True)
    s.add_argument("--type", required=True, choices=TYPES)
    s.add_argument("--subject", default="")
    s.add_argument("--text", default="")
    s.add_argument("--file", default="")
    s.add_argument("--re", default="")
    s.set_defaults(fn=cmd_post)

    s = sub.add_parser("inbox")
    s.add_argument("--all", action="store_true")
    s.add_argument("--last", type=int, default=20)
    s.set_defaults(fn=cmd_inbox)

    s = sub.add_parser("wait")
    s.add_argument("--type", required=True, choices=TYPES)
    s.add_argument("--from", dest="sender", default="")
    s.add_argument("--subject", default="")
    s.add_argument("--any-recipient", action="store_true")
    s.add_argument("--timeout", type=int, default=100)
    s.add_argument("--interval", type=int, default=20)
    s.set_defaults(fn=cmd_wait)

    s = sub.add_parser("status")
    s.add_argument("text")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("board")
    s.add_argument("--last", type=int, default=15)
    s.set_defaults(fn=cmd_board)

    s = sub.add_parser("push")
    s.add_argument("paths", nargs="+")
    s.add_argument("-m", "--message", required=True)
    s.set_defaults(fn=cmd_push)

    s = sub.add_parser("gpu")
    s.add_argument("action", choices=("acquire", "release"))
    s.add_argument("--timeout", type=int, default=100)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_gpu)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
