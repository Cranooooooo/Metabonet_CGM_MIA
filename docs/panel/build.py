# -*- coding: utf-8 -*-
"""生成面板。

    python3 build.py cgm_steps.py index.html            # 只出中文
    python3 build.py cgm_steps.py index.html cgm_steps_en.py   # 中英双语,右上角切换

英文那份是一个结构完全一样的内容文件(同样的 STEPS 形状、同样的数字),
只有文字换成英文。渲染时两份各出一个 .wrap 写进同一个页面。
"""
import importlib.util, os, sys
import render


def load(path):
    spec = importlib.util.spec_from_file_location("content_" + os.path.basename(path)[:-3], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


src = sys.argv[1] if len(sys.argv) > 1 else "example_steps.py"
out = sys.argv[2] if len(sys.argv) > 2 else "index.html"
src_en = sys.argv[3] if len(sys.argv) > 3 else None

m = load(src)
en = None
if src_en:
    e = load(src_en)
    en = dict(title=e.TITLE, short_title=getattr(e, "SHORT_TITLE", None),
              kicker=e.KICKER, lead=e.LEAD, steps=e.STEPS,
              glossary=getattr(e, "GLOSSARY", None))

render.build(m.TITLE, m.KICKER, m.LEAD, m.STEPS, out,
             short_title=getattr(m, "SHORT_TITLE", None),
             glossary=getattr(m, "GLOSSARY", None), en=en)
print(f"{src}{' + ' + src_en if src_en else ''} -> {out}  ({os.path.getsize(out)//1024} KB)")
