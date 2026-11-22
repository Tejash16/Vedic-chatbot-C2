# How the four sessions work together

Four Claude Code sessions run the C2 benchmark. A and B are on Kedar's laptop. C and D are on Tejash's laptop. C is the lead and writes the final results.

Each session has its own clone of this repo. Your letter is in the `.session` file in your clone. All coordination goes through the bus in this repo, on both laptops, so there is one way of working for everyone.

## The bus

Run these from your clone root, with the laptop's virtual environment active.

| To do this | Run |
|---|---|
| Read new messages | `python bus/bus.py inbox` |
| Send a message | `python bus/bus.py post --to C --type QUESTION --subject "short title" --text "your message"` |
| Reply to a message | add `--re <message id>` |
| Wait for an event | `python bus/bus.py wait --type READY --from C --subject B_READY` |
| Say what you are doing | `python bus/bus.py status "embedding the corpus, about 10 minutes left"` |
| See everyone | `python bus/bus.py board` |
| Push your own files | `python bus/bus.py push results/A/results.csv results/A/findings.md -m "A results"` |

Message types: `INFO`, `QUESTION`, `ANSWER`, `READY`, `BLOCKED`, `DONE`. Send to one letter or to `ALL`.

## When to check the bus

- At the start, and before and after every long step.
- While waiting: run the `wait` command. It returns after about 100 seconds if nothing arrived. Run it again. For a long wait, add `--timeout 540` and give the command a 10-minute limit. Do not end your turn while you are waiting for another session.
- Answer any `QUESTION` addressed to you before you carry on with your own work.
- Update your status at every step change.

## Events everyone relies on

| Event | Posted by | Means |
|---|---|---|
| `READY`, subject `B_READY` | C | Shared code, manifest, questions and config are pushed. A, B and D may start |
| `READY`, subject `CANDIDATES_READY` | C | `shared/candidates_top50.jsonl` is pushed. B may run the rerankers |
| `DONE` | A, B, D | That session's `results.csv` and `findings.md` are pushed |
| `DONE` to ALL | C | Final results are written. Everyone stops |

## Git rules

- Write only to your own places: `src/<your letter>/`, `results/<your letter>/`, your status file and your messages. `src/common/` and `shared/` belong to C. If you need a change there, ask C on the bus.
- Push with `bus.py push`. It commits only the files you name, pulls with rebase, and retries. Never force-push. Never run `git add -A`.
- Never commit anything under `../local`, and never commit source text. The `.gitignore` blocks the usual file types. Check `git status` before any manual commit.

## The GPU

Two sessions share one GPU on each laptop.

- Before a GPU job: `python bus/bus.py gpu acquire`. If it says the GPU is busy, run it again until you get it.
- After the job, and also if the job fails: `python bus/bus.py gpu release`.
- Hold the lock only while a job is running. Free GPU memory before you release.

## The two sessions on one laptop

They share `../local`. Files that one session writes there (tables, chunks, embeddings, models) can be read by the other. Do not write the same file from both. Announce on the bus when a shared local file is ready.

## Questions, blocks and the human

- If something is unclear and another session owns it, post a `QUESTION` to that session and carry on with other work if you can.
- If you cannot continue, post `BLOCKED` to C with the reason, and tell your human in the chat as well.
- Anything that needs a person (the meaning of a column, a paid step, a missing file, a login) goes to your human in the chat. The bus is for the sessions.
- If you see `DONE` to ALL from C, finish what you are writing, push it, and stop.
