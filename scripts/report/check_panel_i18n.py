# -*- coding: utf-8 -*-
"""核对面板的中英两份内容文件。

为什么要有这个脚本:英文版是手写的第二份文件,不是自动翻译。手写就会漏 ——
漏一段没翻、漏一个数字打错、少一条结论,页面照样能构建成功、看起来也正常,
只有读英文的人会看到半句中文。所以这三件事必须机器查:

  1. 英文文件里不许出现任何中日韩字符或全角标点
  2. 两份的结构必须完全一样(步骤数、图数、每张图的行数、结论条数)
  3. 每张图里的数字必须逐个相等 —— 翻译只许改文字,不许碰数据

用法(走 0 卡作业,不在登录节点上跑):
    qsub -v TASK=scripts/report/check_panel_i18n.py scripts/pbs/dev/fetch.pbs
"""
import importlib.util, os, re, sys
from collections import Counter

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "panel")

# 中日韩统一表意文字 + 中文标点 + 全角字符
CJK = re.compile(r"[　-〿一-鿿＀-￯‘’“”]")


def load(name):
    p = os.path.join(HERE, name)
    spec = importlib.util.spec_from_file_location(name[:-3], p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def walk_strings(o, path="", out=None):
    """把嵌套结构里的每个字符串连同它的位置一起吐出来,报错时能直接定位。"""
    if out is None:
        out = []
    if isinstance(o, str):
        out.append((path, o))
    elif isinstance(o, dict):
        for k, v in o.items():
            walk_strings(v, f"{path}.{k}", out)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            walk_strings(v, f"{path}[{i}]", out)
    return out


def numbers(o, out=None):
    """只取数字,顺序敏感。bool 是 int 的子类,要单独放行 —— tallyRows 的高亮位是 True。"""
    if out is None:
        out = []
    if isinstance(o, bool):
        out.append(o)
    elif isinstance(o, (int, float)):
        out.append(o)
    elif isinstance(o, dict):
        for k in sorted(o):
            numbers(o[k], out)
    elif isinstance(o, (list, tuple)):
        for v in o:
            numbers(v, out)
    return out


def main():
    zh, en = load("cgm_steps.py"), load("cgm_steps_en.py")
    bad = []

    # --- 1. 英文文件里不许有中文 ---
    scope = [("TITLE", en.TITLE), ("SHORT_TITLE", en.SHORT_TITLE),
             ("KICKER", en.KICKER), ("LEAD", en.LEAD),
             ("GLOSSARY", en.GLOSSARY), ("STEPS", en.STEPS)]
    for name, obj in scope:
        for path, s in walk_strings(obj, name):
            hit = CJK.findall(s)
            if hit:
                bad.append(f"[残留中文] {path}: {''.join(sorted(set(hit)))!r}  ...{s[:60]}...")

    # --- 1b. 渲染层里不许有【写死的】中文 ---
    # 为什么单独查这一段:内容文件干净不等于页面干净。图表是中英两版共用的一份 JS
    # 画出来的,那里写死一句中文就会直接画在英文图上,而上面第 1 项完全看不见它。
    # 外部审查就是这样抓到「训练顺序箭头」四个字出现在四张英文散点图上的。
    # 规矩:JS 里凡是要显示给人看的中文,必须写成 L(svg,'中文','English')。
    src = open(os.path.join(HERE, "render.py"), encoding="utf-8").read()
    js = src[src.index("CHART_JS = r\"\"\""):src.index("PAGE_CSS = ")]
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)          # 注释里的中文不上页面
    ok = set(re.findall(r"L\(\s*svg\s*,\s*'([^']*)'", js))  # 已经配了英文的
    seen = 0
    for lit in re.findall(r"'([^'\n]*)'", js):
        if CJK.search(lit):
            seen += 1
            if lit not in ok:
                bad.append(f"[渲染层写死中文] render.py 的 CHART_JS 里: {lit!r}  "
                           f"—— 改成 L(svg,{lit!r},'English')")
    # 空跑的检查等于没有检查:如果切片或去注释哪一步出错,上面会一条都扫不到而「通过」。
    # 所以这里要求至少看见几条已知的中文字面量,看不见就直接判失败。
    if seen < 3:
        bad.append(f"[检查本身失效] CHART_JS 里只扫到 {seen} 条中文字面量,预期至少 3 条 —— "
                   f"多半是切片范围或去注释的正则坏了,这道检查正在空跑")
    print(f"  渲染层:扫到 {seen} 条中文字面量,其中 {len(ok)} 条已配英文")

    # UI 的英文那一套里也不许有中文,只有语言切换按钮例外
    # (按钮上写「中文」是故意的:切换键一向用目标语言自己的写法标注)
    ui = src[src.index("UI = {"):src.index("def _wrap")]
    en_blk = ui[ui.index('"en"'):]
    for k, v in re.findall(r"(\w+)\s*=\s*\"([^\"]*)\"", en_blk):
        if CJK.search(v) and k != "other":
            bad.append(f"[渲染层写死中文] render.py 的 UI['en'] 里 {k}={v!r}")

    # --- 1c. 正文里的数字必须中英一致 ---
    # 上面第 1、1b 项管的是结构、图里的数字、和有没有漏翻;正文里写的数字它们一个都查不到。
    # 而正文恰恰是最容易改了一边忘另一边的地方 —— 2026-09-01 就发生过:
    # 中文补了「32 格里 31 格高于上限」,英文那边还写着「约一半」。
    #
    # 中英行文本来就有合法的数字差异,硬比会天天误报,误报多了这道检查就没人看。
    # 所以先归一:日期时刻不比、中文「N 万」和英文「Nk」换算到同一个数、
    # 英文数词(sixteen)和中文数词(十个)都还原成数字、整数只比 >= 10 的。
    # 这套规则是拿真实文本调出来的:未归一时 16 处误报,归一后剩 1 处真差异。
    W = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
         "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
         "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
         "nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,
         "seventy":70,"eighty":80,"ninety":90}
    T = {k: v for k, v in W.items() if v >= 20 and v % 10 == 0}
    DEC = re.compile(r"\d+\.\d+")
    _cmp = _withnum = 0

    def _norm(s, lang):
        s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)     # 千分位。必须正好三位,
        s = re.sub(r"\d{4}-\d{2}-\d{2}", " ", s)          # 否则「9/13,10 万步」会被拼成一个数
        s = re.sub(r"\d{1,2}:\d{2}", " ", s)
        s = re.sub(r"\d+\s*月", " ", s)                    # 中文月份,英文写成月份名
        if lang == "zh":
            s = re.sub(r"一万(?=[次条个])", "10000", s)
            s = re.sub(r"十(?=[个次档种条])", "10", s)
            s = re.sub(r"(\d+(?:\.\d+)?)\s*万",
                       lambda m: str(int(float(m.group(1)) * 10000)), s)
        else:
            s = re.sub(r"\bten[- ]thousand\b", "10000", s, flags=re.I)
            s = re.sub(r"(\d+(?:\.\d+)?)k\b",
                       lambda m: str(int(float(m.group(1)) * 1000)), s)
            s = re.sub(r"\b(" + "|".join(T) + r")-(" +
                       "|".join(k for k, v in W.items() if v < 10) + r")\b",
                       lambda m: str(T[m.group(1).lower()] + W[m.group(2).lower()]),
                       s, flags=re.I)
            s = re.sub(r"\b(" + "|".join(W) + r")\b",
                       lambda m: str(W[m.group(1).lower()]), s, flags=re.I)
        return s

    def _nums(s, lang):
        t = _norm(s, lang)
        return (Counter(DEC.findall(t)),
                Counter(x for x in re.findall(r"\d+", DEC.sub(" ", t)) if int(x) >= 10))

    # 导语(现在是一串要点)也要比 —— 它是全页读得最多的一段
    if len(zh.LEAD) != len(en.LEAD):
        bad.append(f"[结构] 导语要点数不同: 中 {len(zh.LEAD)} vs 英 {len(en.LEAD)}")
    for j, (x, y) in enumerate(zip(zh.LEAD, en.LEAD)):
        (da, ia), (db, ib) = _nums(x, "zh"), _nums(y, "en")
        _cmp += 1
        if da or ia or db or ib:
            _withnum += 1
        for kind, ca, cb in (("小数", da, db), ("整数", ia, ib)):
            if ca != cb:
                bad.append(f"[导语{kind}对不上] 要点 {j+1}: "
                           f"仅中文有 {sorted((ca-cb).elements())}  "
                           f"仅英文有 {sorted((cb-ca).elements())}")

    for i, (a, b) in enumerate(zip(zh.STEPS, en.STEPS)):
        fields = [("purpose", a["purpose"], b["purpose"]), ("next", a["next"], b["next"])]
        for key in ("plan", "conclusions"):
            for j, (x, y) in enumerate(zip(a.get(key, []), b.get(key, []))):
                fields.append((f"{key}[{j}]", x, y))
        for fname, x, y in fields:
            (da, ia), (db, ib) = _nums(x, "zh"), _nums(y, "en")
            _cmp += 1
            if da or ia or db or ib:
                _withnum += 1
            for kind, ca, cb in (("小数", da, db), ("整数", ia, ib)):
                if ca != cb:
                    bad.append(f"[正文{kind}对不上] 第 {i} 步 {fname}: "
                               f"仅中文有 {sorted((ca-cb).elements())}  "
                               f"仅英文有 {sorted((cb-ca).elements())}")

    print(f"  正文数字:比对 {_cmp} 段,其中含数字的 {_withnum} 段")
    if _cmp < 50 or _withnum < 20:
        bad.append(f"[检查本身失效] 正文只比对了 {_cmp} 段、{_withnum} 段含数字,远低于预期 —— 这道检查正在空跑")

    # --- 2. 结构必须一样 ---
    if len(zh.STEPS) != len(en.STEPS):
        bad.append(f"[结构] 步骤数不同: 中 {len(zh.STEPS)} vs 英 {len(en.STEPS)}")
    if len(zh.GLOSSARY) != len(en.GLOSSARY):
        bad.append(f"[结构] 名词数不同: 中 {len(zh.GLOSSARY)} vs 英 {len(en.GLOSSARY)}")

    for i, (a, b) in enumerate(zip(zh.STEPS, en.STEPS)):
        tag = f"第 {i} 步"
        if a["status"] != b["status"]:
            bad.append(f"[结构] {tag} status 不同: {a['status']} vs {b['status']}")
        for key in ("plan", "conclusions", "charts"):
            if len(a.get(key, [])) != len(b.get(key, [])):
                bad.append(f"[结构] {tag} {key} 条数不同: "
                           f"{len(a.get(key, []))} vs {len(b.get(key, []))}")
        for k in ("purpose", "next"):
            if not b.get(k):
                bad.append(f"[结构] {tag} 缺 {k}")
        for j, (ca, cb) in enumerate(zip(a.get("charts", []), b.get("charts", []))):
            ct = f"{tag} 图 {j}"
            if ca["type"] != cb["type"]:
                bad.append(f"[结构] {ct} 图型不同: {ca['type']} vs {cb['type']}")
            if ca.get("h") != cb.get("h"):
                bad.append(f"[结构] {ct} 高度不同")
            if len(ca["data"]) != len(cb["data"]):
                bad.append(f"[结构] {ct} 数据行数不同: {len(ca['data'])} vs {len(cb['data'])}")
            # --- 3. 数字必须一样 ---
            na, nb = numbers(ca["data"]), numbers(cb["data"])
            if na != nb:
                bad.append(f"[数字] {ct} 数据对不上\n      中: {na}\n      英: {nb}")
            oa, ob = numbers(ca["opt"]), numbers(cb["opt"])
            if oa != ob:
                bad.append(f"[数字] {ct} opt 里的数字对不上: {oa} vs {ob}")

    if bad:
        print("==== 面板中英核对失败,共 %d 处 ====" % len(bad))
        for x in bad:
            print(" ·", x)
        sys.exit(1)
    print("==== 面板中英核对通过 ====")
    print(f"     {len(zh.STEPS)} 个步骤 · "
          f"{sum(len(s['charts']) for s in zh.STEPS)} 张图 · "
          f"{sum(len(s['conclusions']) for s in zh.STEPS)} 条结论 · "
          f"{len(zh.GLOSSARY)} 个名词 · 两份结构一致、数字一致、英文版无中文残留")


if __name__ == "__main__":
    main()
