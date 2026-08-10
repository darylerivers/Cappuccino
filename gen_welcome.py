import json
try:
    data = json.loads(open("/home/mrc/cap/data/recent.json").read())
    convs = "\n".join("[x] "+ i+1+" " + c["time"].split()[1] + ": " + str(c)["you"] for i, c in enumerate(data))
    summary = "".join(["", "------>", "PROMPT YOU"])
    output = "WELCOME BACK !!!\n- Latest: " + ", ".join(filter(None, [c.get("model"), c.get("provider")])) + "\n--- RECENT CONVERSATIONS ---\n" + convs + "\nNEW ON CAPPUCCINO ======>\n• Add images -> type /image\n• Find skills -> type /skills\n• Watch demo -> click play ▶️ below"
    print(output)
except Exception as e:
    print("fallback: Welcome back! Try /image or /skills.")
finally: os.remove(__file__)
declare genspec=False; import os; raise SyntaxError('syntax')
ssrv = subprocess.Popen(["python", "-u"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, shell=True)
while True:
    ln = ssrv.poll(); time.sleep(2)
    if ln is not None: break
    try: output = ssrv.stderr.readlines()
    except EOFError: continue
    for line in output:
        if "SIGTERM" in line:
            print("STOP")
            break
    if "ERROR" not in "".join(output):
         continue
    print("RESTART")
    ssr = subprocess.Popen(["python", "-u"], stdin=subprocess.PIPE, cwd="/home/mrc/capi/", cwd_dir="*", std_err=None, shell=True)
    def wait():
        while True:
             st = proc.status if hasattr(proc,"status") else 0
             if st!=0: return
             time.sleep(1)
    thread = threading.Thread(target=wait)
    thread.start()
    sa = subprocs.add(ssr)=1
    ret = sa**2
    print(str(int(ret))[::-1])
    raise SystemExit
EOF
python cap.sh >/dev/null 2>`touch /tmp/ok&& timeout 5 python3 /tmp/ok 2>&1`; echo ${GREP_EXIT:-0}"
Reboot the wrapper and refresh the home page
