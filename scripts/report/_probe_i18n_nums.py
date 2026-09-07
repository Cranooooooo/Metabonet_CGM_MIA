# -*- coding: utf-8 -*-
"""探针 v3:先摘小数再取整数,避开「句尾数字后面跟句号」被漏掉的坑。"""
import importlib.util, os, re, collections
HERE = "docs/panel"
def load(n):
    s = importlib.util.spec_from_file_location(n[:-3], os.path.join(HERE, n))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
WORDS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
         "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
         "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
         "nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,
         "seventy":70,"eighty":80,"ninety":90}
TENS = {k:v for k,v in WORDS.items() if v>=20 and v%10==0}
DEC = re.compile(r"\d+\.\d+")
def norm(s, lang):
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)      # 千分位(必须正好三位)
    s = re.sub(r"\d{4}-\d{2}-\d{2}", " ", s)             # 日期
    s = re.sub(r"\d{1,2}:\d{2}", " ", s)                 # 时刻
    s = re.sub(r"\d+\s*月", " ", s)                      # 中文月份
    if lang == "zh":
        s = re.sub(r"一万(?=[次条个])", "10000", s)
        s = re.sub(r"十(?=[个次档种条])", "10", s)
        s = re.sub(r"(\d+(?:\.\d+)?)\s*万", lambda m: str(int(float(m.group(1))*10000)), s)
    else:
        s = re.sub(r"(\d+(?:\.\d+)?)k\b", lambda m: str(int(float(m.group(1))*1000)), s)
        s = re.sub(r"\b(" + "|".join(TENS) + r")-(" + "|".join(k for k,v in WORDS.items() if v<10) + r")\b",
                   lambda m: str(TENS[m.group(1).lower()] + WORDS[m.group(2).lower()]), s, flags=re.I)
        s = re.sub(r"\b(" + "|".join(WORDS) + r")\b",
                   lambda m: str(WORDS[m.group(1).lower()]), s, flags=re.I)
    return s
def pick(s, lang):
    t = norm(s, lang)
    dec = collections.Counter(DEC.findall(t))
    ints = collections.Counter(x for x in re.findall(r"\d+", DEC.sub(" ", t)) if int(x) >= 10)
    return dec, ints
zh, en = load("cgm_steps.py"), load("cgm_steps_en.py")
nd = ni = 0
for i,(a,b) in enumerate(zip(zh.STEPS, en.STEPS)):
    f=[("purpose",a["purpose"],b["purpose"]),("next",a["next"],b["next"])]
    for k in ("plan","conclusions"):
        for j,(x,y) in enumerate(zip(a.get(k,[]),b.get(k,[]))): f.append((f"{k}[{j}]",x,y))
    for name,x,y in f:
        (da,ia),(db,ib)=pick(x,"zh"),pick(y,"en")
        if da!=db:
            nd+=1; print(f"[小数] 第 {i} 步 {name}: 仅中 {sorted((da-db).elements())}  仅英 {sorted((db-da).elements())}")
        if ia!=ib:
            ni+=1; print(f"[整数] 第 {i} 步 {name}: 仅中 {sorted((ia-ib).elements())}  仅英 {sorted((ib-ia).elements())}")
print(f"\n小数不一致 {nd} 处,整数(>=10)不一致 {ni} 处")
