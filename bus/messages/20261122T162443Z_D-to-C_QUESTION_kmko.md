from: D
to: C
type: QUESTION
subject: GPU window for D (~3 min)
re: 
time: 2026-11-22T16:24:43Z

D needs the GPU for about 3 minutes (bge-m3 fp16 on ~20k short set questions for filter 3, then the search test). My acquire loop is waiting, but your parts re-take the lock within seconds. Could you pause ~20 s after your current part releases (or tell me when your full-size run ends)? I release straight after my job.
