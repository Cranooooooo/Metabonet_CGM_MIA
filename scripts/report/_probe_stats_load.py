import ast, sys
for f in ("src/cgmoutlier/generators/igfm.py", "vendor/IG-FM/igfm_core.py"):
    p = "/home/users/industry/imperial/lcheng/workspace/Project_CGM/CGM-OutlierMIA-master/" + f
    try:
        ast.parse(open(p).read()); print(f"{f}: 语法 OK")
    except SyntaxError as e:
        print(f"{f}: 第 {e.lineno} 行 {e.msg}")
