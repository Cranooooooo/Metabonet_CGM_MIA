# -*- coding: utf-8 -*-
"""构建面板并生成给 Artifact 用的片段。整个流程一次跑完,不在登录节点上跑 python。

顺序是有意的:先核对,核不过就不构建 —— 免得一个漏翻的英文页被发出去。

    qsub scripts/pbs/dev/panel.pbs
"""
import os, re, subprocess, sys

PANEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "panel")
PANEL = os.path.abspath(PANEL)


def run(argv, cwd=None):
    print("\n$ " + " ".join(argv), flush=True)
    r = subprocess.run(argv, cwd=cwd)
    if r.returncode != 0:
        sys.exit(r.returncode)


# 1. 中英核对(结构、数字、英文版有没有漏翻)
run([sys.executable, os.path.join(PANEL, "..", "..", "scripts", "report",
                                  "check_panel_i18n.py")])

# 2. 每个数字能不能对回结果文件
run([sys.executable, os.path.join(PANEL, "..", "..", "scripts", "report",
                                  "check_night_numbers.py")])

# 3. 构建双语页面
run([sys.executable, "build.py", "cgm_steps.py", "index.html", "cgm_steps_en.py"],
    cwd=PANEL)

# 4. 结构自检:每一步该有的字段、hero 越界、折线的点数对不对得上 x 轴
sys.path.insert(0, PANEL)
import importlib.util
spec = importlib.util.spec_from_file_location("c", os.path.join(PANEL, "cgm_steps.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
bad = []
for st in m.STEPS:
    need = ("name", "status", "purpose", "plan", "next")
    if st["status"] != "todo":
        need = need + ("conclusions",)
    for k in need:
        if not st.get(k):
            bad.append((st["name"][:14], "缺 " + k))
    for c in st["charts"]:
        o, d = c.get("opt", {}), c["data"]
        if isinstance(o.get("hero"), int) and o["hero"] >= len(d):
            bad.append((st["name"][:14], "hero 越界"))
        if c["type"] == "hairlineLine":
            lens = {len(x["v"]) for x in d}
            if len(lens) != 1 or (o.get("x") and len(o["x"]) not in lens):
                bad.append((st["name"][:14], "折线点数和 x 轴对不上"))
print("\n结构自检:", bad or "全部通过")
if bad:
    sys.exit(1)

# 5. 生成 Artifact 片段。Artifact 会自己套 <!doctype>/<head>/<body>,
#    并且已经给了 charset 和 viewport,所以这两个 meta 要去掉,其余原样保留。
h = open(os.path.join(PANEL, "index.html"), encoding="utf-8").read()
head = h[h.index("<head") + h[h.index("<head"):].index(">") + 1: h.index("</head>")]
body = h[h.index("<body") + h[h.index("<body"):].index(">") + 1: h.index("</body>")]
head = re.sub(r'<meta charset[^>]*>\s*', "", head)
head = re.sub(r'<meta name="viewport"[^>]*>\s*', "", head)
frag = head.strip() + "\n" + body.strip() + "\n"
out = os.path.join(PANEL, "panel_artifact.html")
open(out, "w", encoding="utf-8").write(frag)
print(f"\n片段已生成 {out}  ({len(frag.encode('utf-8'))//1024} KB)")

for s in m.STEPS:
    print(f"  {s['name'][:36]:38s} [{s['status']:8s}] 图 {len(s['charts'])}  结论 {len(s['conclusions'])}")
print("\n==== PANEL OK ====")
