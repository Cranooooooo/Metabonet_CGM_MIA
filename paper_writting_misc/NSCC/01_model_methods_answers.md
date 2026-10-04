# 模型与训练方法：逐条回答

回答对象：`../writing_AI/01_model_methods_questions.md`（写作 AI 的 18 个问题）
整理日期：2026-10-04，依据 NSCC 上的仓库 `CGM-OutlierMIA-master`（HEAD `7a8a22c`）。

**依据范围。** 下面每一条都来自实际产生论文数字的代码、PBS 脚本和各运行目录里的 `meta.json`，不是代码默认值。每条事实都标了出处（`路径:行号` 或运行目录）。确实查不到的，写明「未记录」或「需作者确认」，没有猜。

**三个设置在仓库里的名字**

| 论文 | 仓库 cell | 窗口 T × 通道 C | cohort 目录 |
|---|---|---|---|
| 一天血糖 | `d1_c1` | 288 × 1 | `data/cohort/matrix_d1_c1` |
| 一天血糖＋基础胰岛素 | `d1_c2` | 288 × 2 | `data/cohort/matrix_d1_c2` |
| 七天血糖 | `d7_c1` | 2016 × 1 | `data/cohort/matrix_d7_c1` |

（仓库里还有 `d7_c2`，已停跑，论文不用。）

---

## ⚠️ 先读：写 Methods 前必须改掉或注明的几件事

1. **「同一对模型共用训练种子」是错的。** 现在 `main.tex` 第 519–527 行（Paired design）写着 *"Both members of a pair share the generator, its hyperparameters and its training seed"*。代码并不这样做：种子按**作业名**派生，base 和 include_t 名字不同，所以**种子也不同**（见 Q11）。共用的是**派生规则**，不是种子值。这句话需要改。
2. **七天格子的「本文模型」用的是缩减预算：12,000 步，一天格子是 25,000 步。** 12,000 是看质量曲线（Context-FID）选的，而且是出于省算力。理由和随之而来的局限见 Q9，Methods 里必须写明。
3. **七天格子的 base 模型不是单独训练的。** 它是从一个 25,000 步运行的第 12,000 步存档**摆进来的**（staged），那个源运行续过训、随机数没有接上。对**暴露计数**没有影响（零假设里 base 被消掉了），但**七天的 FID 数字 0.126 就是从这个 base 的释放上算的**。详见 Q12。
4. **两个隐私模块在两个轴上都没有可测的增益。** 改过的模型和原版 backbone 在 FID 和暴露上都分不出来（Q15）。现在的稿子已经如实报了这个 null，Methods 别再写成贡献。
5. **backbone（IG-FM）的引用还没定。** 它来自作者自己的仓库 `Cranooooooo/IGFM_ICDE2027`，看名字是投 ICDE 2027 的工作。论文用什么引用（已发表？预印本？「under review」？），**需要作者确认**（见 Q2、Q18）。

基线那边还有三件要在 Methods 里如实写的事，细节见 Q13–14：

6. **预算并没有完全对齐。** DiM-TS 实际喂了 12.8 M 个序列，是其它臂（约 6.4 M）的两倍，因为有写死的梯度累积 2。
7. **容量没有对齐。** 以本文模型为 1×，基线是 0.11×～5.0×。三个新基线都用适配器默认值，没有调参。
8. **FourierDiffusion 跑的是时域版本**，频域那一部分被关掉了。

另外看到一处数据描述和 `docs/DATA.md` 对不上，顺带报：`main.tex` 第 476 行写 "1,291 participants across twelve source studies"。DATA.md 记录的是 **14 个**来源研究；按 (study, id) 重新编键后是 **1,907 个**受试者（1,291 是按 id 单独计、把不同人混在一起的数）。这一段归数据，不归模型，请写作方核对。

---

# 一、我们的模型是什么

## Q1. 「our model」对应哪个版本

**名称**：IG-FM（Imputation-Guided Flow Matching）＋ 两个隐私模块（仓库里叫「改动一」coord_mask、「改动二」level pooling）。论文表里写作 **"This work"**，`membership_null.py` 里的键是 `ours`。

**代码位置**

| 部件 | 文件 |
|---|---|
| 网络、损失、采样器（含两个模块） | `vendor/IG-FM/igfm_core.py` |
| 未改动的上游原版（对照用） | `vendor/IG-FM/igfm_core.py.stock` |
| 训练循环、数据接入、存档/续训 | `src/cgmoutlier/generators/igfm.py` |
| 每个作业训一个模型、放一组样本 | `src/cgmoutlier/loo/train.py`，入口 `scripts/run_loo.py` |
| 实际生效的超参（三个格子共用的底） | `scripts/pbs/dev/igfm_priv.params` |

**代码版本**（仓库 `github.com/Cranooooooo/Metabonet_CGM_MIA`）

- 两个模块写进 `igfm_core.py`，提交于 **`181c0ca`**（2026-09-07）。但一天的两个格子是 **2026-09-01～02** 用当时的工作区跑的（`igfm_core.py` 的 mtime 是 09-01 20:12，比 `loo_igfm_priv_d1_c1/base` 开训早几分钟），比这次提交早。之后这个文件只改过一次（`ac13fa2`），而且只改了续训逻辑。所以 `181c0ca` 里的模型代码就是一天格子实际跑的代码。**严格说，这些运行没有记录 commit hash**，`meta.json` 里没有这个字段。
- 七天格子的 26 个 include 模型是 2026-09-14 之后训的，用的是 **`ac13fa2`**（同一份模型代码，加了「续训时恢复随机数状态」）。

**三个设置是否同一版本**：网络和损失代码相同，超参底也相同（`igfm_priv.params`）。只有以下几处不同，都写在各树的 `.params.json` 和 `meta.json` 里：

| | d1_c1 | d1_c2 | d7_c1 |
|---|---|---|---|
| 训练步数 | 25,000 | 25,000 | **12,000** |
| 微批 × 累积 | 32 × 8 | 32 × 8 | **8 × 32** |
| 有效 batch | 256 | 256 | 256 |
| 参数量 | 2.09 M | 2.09 M | **2.34 M**（多出的是位置编码） |

## Q2. 原始 backbone 是什么

**backbone 就是 IG-FM**，从作者自己的仓库 `Cranooooooo/IGFM_ICDE2027` 的 commit **`3f9a246`** 拷来（2026-08-02），上游文件 `IG_FM/train_r10.py`（出处 `vendor/IG-FM/PROVENANCE.md`）。上游 README 写的是 "IG-FM (Imputation-Guided Flow Matching for Multivariate Time-Series Generation)"，ICDE 2027。**这篇的发表状态和引用格式需要作者提供。**

**直接沿用**（逐字，见 `igfm_core.py.stock`）：

- VelocityNet 整体结构（编码器–瓶颈–解码器 Transformer）
- 双分支训练（生成分支＋填充分支，IDG soft prior）
- 损失：L_gen、L_imp、L_decor、L_corrmap，以及 λ 调度
- AdamW、EMA、Euler 采样器

**为 CGM 改的，不是选择而是被数据逼的**（`igfm.py` 文档字符串）：

1. **不用上游的数据加载器。** 上游从一个连续 CSV 里按步长 1 切窗。我们的窗口是不同受试者、互不重叠、不含缺口的整天（或整周）。拼起来再切会造出跨天、跨人的窗口，所以直接喂预先切好的窗口。
2. **有效 batch 256 不变，改用微批＋梯度累积凑出来。** 注意力是 O(T²)，T=288 时放不下 256 的整批。
3. **宽度改成 hidden=144。** 上游各数据集用 128–512。144 是为了让参数量落在约 2 M（2.09 M），对齐基线的容量，见 Q14。
4. **采样步数从 200 改成 500**，和 DiM-TS 参照集的口径一致（`igfm_pilot.pbs` 第 3 条）。

**我们新增的**（`diff igfm_core.py.stock igfm_core.py`；全部有开关，全关时和原版逐位一致）：

- 改动一 `coord_mask`：按坐标遮挡（Q4）
- 改动二 `level_dims` / `order_dims`：瓶颈里的水平子空间沿时间池化，外加分组去相关（Q4）
- 改动三 `iso_weight`：按窗口孤立度加权。**写了，但所有报告的运行里都关着**（`iso_weight=false`）。原因：训练步拿不到受试者编号，排除不了「同一个人的其它窗口」，部署的统计量和验证过的那个对不上（`igfm_core.py` 中 `isolation_weights` 的注释）。论文不应提它，或者只作为「已放弃」一句带过。

## Q3. 完整网络结构

模型定义：`vendor/IG-FM/igfm_core.py` 的 `class VelocityNet`（原版在 `igfm_core.py.stock` 第 140–200 行）。

**输入**

| 输入 | 形状 | 含义 |
|---|---|---|
| x_t | [B, T, D] | 流时间 t 上的带噪状态 |
| t | [B] | 流时间，∈[0,1] |
| M_cond | [B, T, D] | 条件掩码 |
| X_cond | [B, T, D] | = M_cond · x0 |
| X_prior | [B, T, D] | 生成分支给的软先验，可以为空 |

生成时后三项全是 0，D = 通道数（1 或 2），**每个时间步是一个 token**，通道在输入投影里混合。

**前向流程**

```
h = Linear_in(x_t) + Linear_mask(M_cond) + Linear_cond(X_cond)  [+ Linear_prior(X_prior)]
      # cond / prior 两个投影无偏置、零初始化
h = h + pos_emb[:, :T]                       # 可学习位置编码 [1, T, 144]，trunc_normal(std 0.02)
h = h + MLP(Sinusoidal(t))                   # 正弦时间编码(144) → Linear-SiLU-Linear(144)，按 token 广播相加
z = Encoder(h)                               # 4 层
z = LayerNorm(z)                             # 瓶颈；改动二在这里把前 48 通道沿时间池化（Q4）
h = Decoder(z)                               # 4 层
v = Linear_out(LayerNorm(h))                 # → [B, T, D]，速度场
```

- **每一层**都是 `nn.TransformerEncoderLayer`：d_model 144，8 头（每头 18 维），FFN 576，GELU，pre-norm，dropout 0。解码器同样只有自注意力，没有交叉注意力。
- **参数量**（训练日志 `[igfm] VelocityNet ... params`）：

| 设置 | 参数量 | 说明 |
|---|---|---|
| T=288，D=1 | 2.09 M | |
| T=288，D=2 | 2.09 M | 输入/输出投影多几百个参数，四舍五入后不变 |
| T=2016，D=1 | 2.34 M | 多出的 (2016−288)×144 = 248,832 个全是位置编码 |

- 两个模块**不增加参数**。

## Q4. 结构化 masking 和 level pooling 的实际实现

两个模块**都用于**「This work」那一行的全部三个格子。参数：`coord_mask=true, p_block=0.35, p_level=0.25, level_dims=48, order_dims=48`，出处是各树的 `.params.json` 和每个模型的 `meta.json`。

### 改动一：按坐标遮挡（`coord_mask`）

代码：`igfm_core.py` 的 `sample_cond_masks_coord`、`strip_level`，以及 `training_step` 里 `task == "level"` 那一段。

- **作用在哪**：**只**作用于训练时的**填充分支**。生成分支照旧无条件；**采样时完全不起作用**（采样是无条件的，M_cond = X_cond = 0）。
- **怎么抽**：**每个微批**抽一次任务类型，整批共用：

| 任务 | 概率 | 遮挡对象 | 比例 / 长度 |
|---|---|---|---|
| `block` | 0.35 | 一段**连续时间**，所有通道一起遮 | 段长 = round(T·r)，r ~ U(0.15, 0.55)；每个样本**各自**抽起点（整批同起点会让位置变成捷径） |
| `level` | 0.25 | 逐格 Bernoulli | 比例 ~ U(0.30, 0.80)；然后把条件里的**整体水平**减掉（见下） |
| `cell` | 0.40 | 原版逐格 Bernoulli | 以 0.4/0.9 的概率比例 ~ U(0.10, 0.30)，否则 ~ U(0.30, 0.80) |

  段长换算：T=288 时 43–158 个点（约 3.6–13.2 小时）；T=2016 时约 302–1109 个点（约 1.05–3.85 天）。
- **level 任务怎样「去掉水平」**：对每个样本、每个通道，在**被观测到的格子上**求均值 μ，再从 X_cond 里减掉。为了不留后门，X_prior 也减**同一个** μ（`strip_level` 的注释写了理由）。
- **实现上的局限，写 Methods 时请别夸大**：水平只从**条件通道**里拿掉了。带噪输入 x_t = (1−t)·x0 + t·ε 本身仍然带着水平，t 小的时候尤其明显。所以这个任务逼模型「从条件里推不出水平」，**并不能保证模型拿不到水平**。
- **训练日志里的可证伪读数**（`logs/igfm_priv_loo_shard0.log`，第 25,000 步，按任务的填充误差）：

  | 任务 | 填充误差 L_imp |
  |---|---|
  | block | 0.0655 |
  | level | 0.0227 |
  | cell | 0.0165 |

  level 比 cell 高约 1.4 倍，说明去掉水平这一步确实让任务变难了，但幅度不大。

### 改动二：瓶颈的水平子空间池化（`level_dims`）＋分组去相关（`order_dims`）

代码：`igfm_core.py` 的 `pool_level_subspace`、`decor_loss_groups`，以及 `VelocityNet.forward` 里瓶颈那一段。

- **池化**：瓶颈 z（144 通道）的**前 48 个通道**沿时间取均值，再广播回每个时刻。这一步在**解码器之前**，在**每一次前向**里都做，所以训练的两个分支和采样时都生效。效果是这 48 个通道对每个时刻是同一个数，**结构上只能表达窗口整体的水平**，装不下逐点痕迹。
- **重组**：没有专门的重组步骤。解码器直接吃拼接后的 [池化后的 48 维 ‖ 其余 96 维 原样]。
- **「shape / order」怎么定义**：**没有结构上的定义。** `order_dims=48` 只是给去相关损失划组：0–47 叫「水平」，48–95 叫「时序」，96–143 叫「细节」。损失只罚**组与组之间**的相关（平方均值），组内不罚。它取代了原版对全部非对角元素的去相关。时序组和细节组本身**没有任何结构约束**，所以不能写成「模型把水平和形状分离到不同子空间」。
- **一个细节**：去相关损失用的是**池化前**的 z。池化后那 48 维沿时间是常数，z-score 会除以 0。损失还会先沿时间减均值，**所以它罚不到池化真正保留的那个量**，真正起作用的是池化这道硬容量限制（`decor_loss_groups` 注释）。

### 这些模块是否用于当前结果

- 「This work」那一行：**两个都开**，三个格子都是。改动三全关。
- 「Backbone, unmodified」那一行：三个都关，用的是原版去相关损失。只有 d1_c1、d1_c2 有，d7_c1 没训原版。

## Q5. 模型接收哪些输入／条件

- **训练**：只有窗口本身（标准化后的 [T, C] 数组）。d1_c2 是血糖和每公斤体重的基础胰岛素两个通道，**联合生成**，不是以胰岛素为条件生成血糖。
- **生成**：**无条件生成**，只需要高斯噪声。
- **不使用**：患者身份或受试者编号、临床统计量、窗口均值、人口学信息、日期或时刻、来源研究、设备型号、餐食和推注数据。
- **唯一的「位置」信息**是可学习的位置编码。它编码的是**窗口内**的时间步，不是钟点。窗口按日历日对齐，所以第 k 个位置大致对应一天里的同一时刻。
- 填充分支训练时的掩码 M_cond 和观测值 X_cond 来自同一个窗口，不涉及其它信息。

---

# 二、怎样训练和生成

## Q6. 训练目标

代码：`igfm_core.py` 的 `training_step`（原版在 `igfm_core.py.stock` 第 267–333 行）。每步对同一批 x0 做两次前向。

**生成分支（无条件 FM）**

- ε ~ N(0, I)，t ~ U(0, 1)，**每个样本各抽一个 t**
- 线性路径：x_t = (1−t)·x0 + t·ε，**t=0 是数据，t=1 是噪声**
- 预测目标是速度：v* = ε − x0
- 前向时 M_cond = X_cond = 0，X_prior 不给
- L_gen = 所有格子上 ‖v̂ − v*‖² 的均值
- 由 v̂ 反推 x̂0 = x_t − t·v̂，**截断梯度**后作为下一个分支的软先验

**填充分支（条件 FM）**

- 用**另一组** ε、t
- 掩码按 Q4 抽（原版是 cell，加了改动一以后是 block / level / cell 三选一）
- X_cond = M_cond · x0；X_prior = (1 − M_cond) · x̂0
- L_imp = 只在目标格（被遮的格子）上 ‖v̂ − v*‖² 的均值

**辅助损失**（作用在填充分支的瓶颈 z 上）

- L_decor：原版对全部 144 通道的非对角相关做惩罚；本文模型换成分组版（Q4）
- L_corrmap：x0 和 x̂0 各自与 z 的互相关图之间的 MSE

**总损失**

```
L = L_imp + λ(it)·L_gen + 0.1·L_decor + 0.1·L_corrmap
```

λ 的调度是 `"100000:1,200000:2,300000:4"`。我们最多只训 25,000 步，**第一个边界从来没到过，所以整个训练里 λ ≡ 1**（日志每 5,000 步打印 `lam=1.0`）。

## Q7. 预处理和逆变换

代码：`scripts/build_cohort_multi.py` 第 230–275 行；常数记在各 cohort 的 `manifest.json` 里。

**窗口**

- 一天窗口 = 一个日历日 288 个 5 分钟格，**必须全部存在**。不插值，不完整的天直接丢。
- 七天窗口 = 7 个**连续**的完整日拼成 2016 个点，seams = 0（`matrix_d7_c1/manifest.json`）。

**单位**

| 通道 | 单位 |
|---|---|
| 血糖 | mg/dL |
| 基础胰岛素 | 原始值**除以受试者体重（kg）**（cohort 来自 `metabonet_sid_c2_perkg`，常数完全吻合） |

基础胰岛素原始列的均值约 0.085 / 5 分钟格，按「U / 5 分钟」理解约合 1 U/h，量级合理。**但原始单位我们只从数值上推断，请按 MetaboNet 数据字典核对后再写进论文。** 体重在数据里是磅，构建时换成了 kg（`--per-kg` 的帮助文字）。

**标准化**：逐通道 z-score，裁到 ±5 SD，再除以 5，落在 [−1, 1]。**没有窗口级归一化。**

**常数**

| 格子 | 通道 | 均值 | SD |
|---|---|---|---|
| d1_c1、d7_c1 | 血糖 | 145.937 mg/dL | 57.654 mg/dL |
| d1_c2 | 血糖 | 145.552 | 57.537 |
| d1_c2 | 基础胰岛素 | 1.3649e-3 | 1.5984e-3 |

- d7_c1 **原样继承** d1_c1 的常数，只是把同一批标准化后的值重新分组，没有重新标准化。
- **常数是在哪群人上算的**：在**筛到 506 人之前的母 cohort** 上，用全部窗口算。血糖常数和 `metabonet_sid_c1`（1,329 人，`docs/DATA.md` 里的 145.94 / 57.65）一致；双通道常数和 `metabonet_sid_c2_perkg`（1,208 人）逐位一致。也就是说，常数用到了包括 26 个目标在内的整个母 cohort。这对成对比较是中性的（base 和 include 用同一套常数），但严格说标准化步骤见过所有人，可以在 Methods 里带一句。

**逆变换**：`raw = x · 5 · sd + mean`（manifest 里的 `denorm_formula`），用 `cgmoutlier.data.cohort.to_mgdl`，必须传对应 cohort 的 manifest。

- 生成样本**不做裁剪**，可能超出 [−1, 1]（例如 d7_c1 base 释放的范围是 [−1.97, 2.78]）。逆变换后就是对应的 mg/dL 或每公斤胰岛素值。
- 临床指标是逆变换到 mg/dL 之后算的。
- 基础胰岛素逆变换后出现负值时有没有截到 0，**我没查到专门的处理，需要确认**（只影响效用指标，不影响攻击）。

## Q8. 实际生效的训练参数

| 项目 | d1_c1 | d1_c2 | d7_c1 | 出处 |
|---|---|---|---|---|
| 优化器 | AdamW | 同左 | 同左 | `igfm.py` fit() |
| 学习率 | 5e-4，**常数，没有调度器** | 同 | 同 | 同上；d7 base 的 meta 里 `equivalence_note` 也写了没有 LR scheduler |
| 权重衰减 | 1e-4 | 同 | 同 | |
| 梯度裁剪 | 全局范数 1.0 | 同 | 同 | |
| EMA | decay 0.999，每个优化步都更新（`torch.optim.swa_utils.AveragedModel`） | 同 | 同 | |
| 有效 batch | 256 | 256 | 256 | |
| 微批 × 累积 | 32 × 8 | 32 × 8 | 8 × 32 | `.params.json`；d1_c1 没写，用适配器默认值 32 |
| 抽样方式 | 有放回均匀抽窗口，`torch.randint`，不用 DataLoader，不分 epoch | 同 | 同 | |
| 优化步数 | 25,000 | 25,000 | **12,000** | |
| 等效 epoch（base，5,741 窗口） | ≈ 1,115 | ≈ 1,115 | ≈ 535 | 步数 × 256 / 5,741 |
| 损失权重 | λ_gen = 1，decor 0.1，corrmap 0.1 | 同 | 同 | |
| IDG 掩码参数 | p_pure_gen 0.10，p_cond_low 0.40，low U(0.1,0.3)，high U(0.3,0.8) | 同 | 同 | |
| 精度 | **FP32**，代码里没有 autocast、AMP 或 bf16 | 同 | 同 | grep 结果为空 |
| 存档 | 每 2,000 步存一次，留最新 2 个 | 同 | 每 500 步，留 2 个 | |

**完整 params 字符串**，就是各树 `results/runs/loo_igfm_priv_<cell>/.params.json` 的内容：

```
d1_c1: {"hidden":144,"level_dims":48,"order_dims":48,"steps":25000,"sampling_steps":500,"save_every":2000,"keep_checkpoints":2,"coord_mask":true,"p_block":0.35,"p_level":0.25,"iso_weight":false}
d1_c2: 同上 + "micro_batch":32
d7_c1: 同上 + "micro_batch":8, "steps":12000, "save_every":500   （JSON 里后出现的键覆盖前面的）
```

**25,000 步是怎么来的**（`scripts/pbs/dev/igfm_pilot.pbs` 第 2 条）：按**喂进去的样本数**对齐 DiM-TS 的注册参照点。DiM-TS 是 100,000 步 × batch 64 = 640 万个窗口，640 万 / 256 = 25,000 次迭代。不是调出来的。

## Q9. checkpoint 怎样选、有没有验证集、有没有按结果选过

- **checkpoint 规则**：取**最后一步**的 **EMA 权重**（`IGFMGenerator.sample` 用的是 `self._ema.module`）。没有「最佳验证损失」这种选法。
- **验证集**：**没有。** 不存在留出受试者的验证集。训练集就是作业指定的那些受试者的全部窗口：base 是 475 人 / 5,741 个窗口，include_t 再加上目标 t。
- **质量关**：在 `configs/experiment.yaml` 第 439–473 行，**2026-08-28 注册，早于 IG-FM 的实现**。及格线是「不比同格子的 DiM-TS base 差」，看 Context-FID 和判别分数。它只决定一个模型能不能进表，不用来挑 checkpoint。注意 Context-FID 的真实参照是 **base 的训练窗口**（`quality_final/*.json` 里 n_real = 5,741），**不是留出数据**。

**哪些选择看过什么**

| 选择 | 依据 | 看没看隐私结果 |
|---|---|---|
| hidden=144 | 对齐约 2 M 参数（Q2） | 没看 |
| 一天 25,000 步 | 按样本数对齐 DiM-TS（Q8） | 没看 |
| 两个模块的超参 p_block、p_level、level_dims 等 | **按设计一次定下**，没有扫描；唯一的检验是 d1_c1 base 过质量关（0.0383，原版是 0.0394） | 没看 |
| **七天 12,000 步** | **看过 Context-FID 曲线**（下面展开） | 没看 |

**七天 12,000 步的来历**：一个 25,000 步的 base 存下了全部里程碑；在 2k / 6k / 12k / 18k / 25k 五个点上重新采样，在一次调用里打分（`results/matrix/curve/d7_c1_quality.json`）：

| 步数 | 2,000 | 6,000 | 12,000 | 18,000 | 25,000 |
|---|---|---|---|---|---|
| Context-FID | 13.35 | 0.609 | **0.1265** | 0.1093 | 0.1035 |

25,000 步本身也能过关。选 12,000 是**省算力的决定**：26 个 include 一共省约 1,404 GPU 小时（`docs/PAPER_PLAN.md` 第 754–763 行）。

**这必须作为局限写进论文**：七天这一列里，只有本文模型是在缩减预算下读的，和其它臂的预算不对齐。而且项目自己的数据显示，FID 平的区间里成员风险可以大幅摆动：DiM-TS d1_c1 在 20k→100k 步之间 FID 0.060→0.083，几乎不动，arm AUC 却走了 0.633→0.840→0.515。所以「FID 在 12,000 以后很平」不能推出「泄漏在那里也平」。

**隐私结果有没有反过来影响模型或超参选择**：**没有。** 攻击统计量是 2026-08-07 在 copy-paste 上冻结的；质量门槛在 IG-FM 之前就注册了；预算的依据也都写在上面。

**有一个方向上的先验需要说明**：两个模块本身，是看了 DiM-TS 的泄漏分解（当时的读法是「水平＋时序承载泄漏」）之后**针对它设计的**。后来在新零假设下，这个分解被修正为「时序＋低频承载，水平几乎不承载」。

## Q10. 生成过程

代码：`igfm_core.py` 的 `sample_unconditional`；适配器的 `IGFMGenerator.sample`。

| 项目 | 内容 |
|---|---|
| 初始噪声 | x ~ N(0, I)，形状 [b, T, D] |
| 求解器 | **显式 Euler**，固定步长，时间网格 `linspace(1, 0, 501)`，所以是 **500 步**，从噪声积到数据 |
| 容差 | 无，不是自适应求解器 |
| 每一步 | x ← x − v̂(x, t)·Δt；条件全为 0，无条件生成 |
| 权重 | **EMA** |
| sample_batch | 64（适配器默认值） |
| 后处理 | **无**：不裁剪、不取整、不做任何修正。输出留在标准化空间，存成 `samples.npy`（float32，[K, T, C]） |
| 释放数量 K | **等于该模型自己的训练窗口数**：base 5,741；include_t 是 5,741 + 目标 t 的窗口数（`loo/train.py`，K 默认 = N）。攻击端在比距离之前会把两边裁到同样大小（`attack.statistic._match`） |
| 三个设置 | 完全相同，只有 T、D 不同 |
| 随机性 | 采样**不重新设种子**，接着训练结束时的 RNG 状态往下走（Q11）。七天 base 是例外（Q12） |

## Q11. 随机种子和训练重复

- **每个配置只训了一次**（设计重复 `rep1`）。没有多种子重复。
- **种子怎样派生**（`src/cgmoutlier/loo/train.py` 第 42–50 行）：

  ```
  job_seed = SHA1("2026|<作业名>") 的前 4 个字节，按大端解成整数
  ```

  作业名是 `base` 或 `include_<study>__<id>`，基础种子 `--seed 2026` 由 PBS 传入。
  - 所有格子的 base 都是 `job_seed = 3742878366`。
  - include_DCLP3__10 是 758182966。
- **job_seed 实际控制什么**：适配器 `fit()` 一开头就 `torch.manual_seed(job_seed)` 和 `np.random.seed(job_seed)`（`igfm.py`）。之后所有随机过程都走这条流：权重初始化、每个微批抽哪些窗口（CUDA 上的 `torch.randint`）、两个分支的 ε 和 t、掩码类型与比例、块的起点、Bernoulli 掩码。**采样不再设种子**，初始噪声接着训练结束时的 CUDA RNG 状态往下抽，所以**同一个 job_seed 同时决定训练和采样**。
- **base 和各个 target 模型的种子不同。** 种子由名字决定，**每个模型一个，彼此不同**（27 个 meta 里有 27 个不同的 job_seed）。这样做是故意的：加作业、换顺序都不会改变别的作业的抽样（文档字符串里引的是 `docs/PITFALLS.md` §5）。**所以稿子里「pair shares its training seed」那句要改**，比如改成：*"each model's seed is derived deterministically from its job name, so every model has a distinct, reproducible random stream"*。
- **跨格子、跨对照树种子相同。** 同一个作业名在 d1_c1、d1_c2、d7_c1 里，以及在本文模型（`loo_igfm_priv_*`）和原版 backbone（`loo_igfm_*`）树里，种子都一样。所以 backbone 和本文模型是**同种子配对**的，唯一的差别就是两个模块。
- **基线**：FourierDiffusion、DiffWave、Diffusion-TS、copy-paste 用的也是 job_seed。**DiM-TS 例外**：它 27 个模型的训练种子全是 2026（`dimts.py:115`）。所以「一对模型共用训练种子」这句话，在整篇论文里只对 DiM-TS 成立。
- **结论在不同种子下是否稳定，没有直接测过。** 新零假设本身给了一个替代证据：25 个非成员模型各用各的种子，构成了「同配置、不同种子、不同一个人」的分布。

## Q12. 续训与重新采样

**一天格子（d1_c1、d1_c2，本文模型和原版都算）**

- 108 个模型**都没有续训，也没有重新采样**。meta 里 `fit_resumed` 不存在或为 false，也没有 `resampled_from` 字段。
- 每个模型一个进程训完，接着采样。
- 每个运行目录留着 `iter-24000.pt`、`iter-25000.pt`（含 model / EMA / optimizer / RNG）和 `samples.npy`。
- 例外：`loo_igfm_priv_d1_c1/base` 和 `loo_igfm_priv_d1_c2/base` 是从 `igfm_priv_d1_c*/base` **整份拷进来**的（只拷 samples.npy 和 meta.json，没有拷存档；存档留在源目录）。没有重新训，也没有重新采样。

**七天格子（d7_c1），两类情况**

1. **26 个 include 模型**：其中 **18 个续过训**（24 小时墙钟要接 2–3 次），**18 个都恢复了随机数状态**（`fit_rng_restored: true`）。
   - 续训恢复的内容：模型、EMA、优化器、torch CPU RNG、numpy RNG、全部 CUDA RNG（`igfm.py` fit() 的续训段）。
   - **没有学习率调度器**，λ 也是常数，所以没有别的状态要恢复。
   - 续训是否逐位等价已经测过：`scripts/check_resume_equivalence.py`，A vs B 差 0.000（`d7c1_cleanbase.pbs` 注释）。
   - 训练耗时用 `fit_seconds_total` 读，它跨续训累加。
2. **d7_c1 的 base**（`loo_igfm_priv_d7_c1/base`）**不是**单独训练的：
   - 它是 `scripts/stage_base_from_curve.py` 从 `results/runs/igfm_priv_d7_c1/base/iter-12000.pt` **摆进来的**。那是一个 25,000 步的运行，在第 10,000 步续过训，**当时随机数还没接上**，第 10,000 步之后重放了从第 0 步开始的批次流。
   - 它的释放集由 `igfm_budget_curve.py` 在采样前**重新设种子**抽出来。
   - meta 里如实记着：`source_stream_replayed: true`、`bit_reproducible: false`，以及 `equivalence_note`、`stage_note` 两段说明。
   - **对论文数字的影响**：
     - **暴露计数不受影响。** `membership_null.py` 的经验 p 值是拿 d(base) − d(include_t) 和 d(base) − d(include_j) 比排名，base 项**两边相消**。
     - **七天 FID 0.126** 就是这个 base 释放算的。
     - 「已发表口径」的 gap 受 base 偏移影响：中位数约 −0.0013，按旧口径能把符号翻过来。这正是论文改用非成员均值做参照的原因。
     - 曾计划用 55 GPU 小时重训一个干净的 base（`results/runs/d7c1_cleanbase`），**后来取消了**，目录是空的。理由见 commit `a8bdb31`：新零假设已经把 base 从统计量里移除了。

**不同采样设置或不同版本**：报告的数字全部是 500 步采样、EMA 权重、K = 训练集大小。仓库里还有 st50 / st100 / st200 的采样实验，**只用于 DiM-TS 的采样步数敏感性分析**（`results/matrix/quality/*_st*`），不进主表。

---

# 三、基线与消融怎样比较

## Q13–Q14. 各基线的来源与配置

所有基线都通过同一个驱动 `loo/train.py` 运行：同一个设计，每个格子 27 个模型（base + 26 个 include）；K = 训练集大小；都没有续训、没有重新采样；输入和本文模型一样，是 [−1,1] 标准化后的窗口。每个运行树里 27 个 meta.json 的 params 完全一致（只有 n_samples、workdir 不同）。PBS 正文在 `scripts/pbs/dev/baseline_base.body.sh`（base）和 `baseline_loo.body.sh`（include）；DiM-TS 走 `scripts/pbs/B2_matrix_train.pbs`，由 `launch_matrix.sh` 发起。

**出处**

| 基线 | 论文 | 代码来源 | 锁定的 commit |
|---|---|---|---|
| DiM-TS | Yao, Zuo, Zhang. *DiM-TS: Bridge the Gap between Selective State Space Models and Time Series for Generative Modeling*. arXiv:2511.18312, 2025（`vendor/DiM-TS/README.md`） | github.com/yzh8221/DiMTS；从本地工作副本拷来（`vendor/DiM-TS/PROVENANCE.md`） | **未记录** |
| FourierDiffusion | Crabbé, Huynh, Stanczuk, van der Schaar. *Time Series Diffusion in the Frequency Domain*. arXiv:2402.05933, 2024（ICML 2024，请核对） | github.com/JonathanCrabbe/FourierDiffusion | **未记录** |
| DiffWave | 仓库里没写论文。通常引 Kong et al., *DiffWave: A Versatile Diffusion Model for Audio Synthesis*, ICLR 2021，**需作者确认** | `philsyn/DiffWave-unconditional`（MIT） | **未记录** |
| Diffusion-TS | Yuan & Qiao. *Diffusion-TS: Interpretable Diffusion for General Time Series Generation*. ICLR 2024 | github.com/Y-debug-sys/Diffusion-TS | **未记录** |

四个基线都没有记录上游 commit，`vendor/` 是在 `34b5482`（2026-08-16）一次性整体提交进来的。

**为 CGM 做的适配**

- **DiM-TS**（`vendor/DiM-TS/cgm_train_sample.py`、`src/cgmoutlier/generators/dimts.py`）：`input_shape=[T,C]`，`auto_norm=False`。跨通道相关损失 `mmd_alpha` 在 C=1 时没有定义，所以 C=1 设为 0，C=2 设为 0.0008。没有改结构。因为 Mamba 内核的依赖，它在单独的 conda 环境里以子进程方式运行。
- **FourierDiffusion**（`fourier_diff.py`）：
  - ⚠️ **频域那一部分是关掉的**：`fourier_transform=False`、`fourier_noise_scaling=False`，扩散在**时域**进行，注释写的理由是「为了可比」。这一点 Methods 里必须写明，否则审稿人会认为我们用的是原方法。
  - 适配器自己做逐 (t, c) 的标准化，采样后再逆回去。
  - 关掉了验证循环。
  - 加了一道「塌缩」检查：拿 DSM 损失和零分数模型比，比值大于 0.5 就判失败。
  - 两处写死的 512 窗口前向改成了分块计算，只影响 T=2016，而七天没有跑 FourierDiffusion。
- **DiffWave**（`diffwave.py`）：`in/out_channels = C`，数据整理成 (B, C, T)。厂商代码只改了一处：去掉写死的 `.cuda()`。
- **Diffusion-TS**（`diffusion_ts.py`）：内存数据集，seq_length = T，feature_size = C。另外加了存档轮换、续训修复和临时目录修复。没有改结构。

**实际生效的训练与采样配置**（出处是各运行 `base/meta.json` 加适配器代码，以 d1 为准）

| | DiM-TS | FourierDiffusion | DiffWave | Diffusion-TS |
|---|---|---|---|---|
| 实际传入的 params | `hidden_size 256, steps 100000, batch 64`，梯度累积 2（写死） | `max_epochs 1115, batch 64, keep_best false` | `total_iters 100000, batch 64` | `max_epochs 100000`（实为迭代数）`, batch 64, grad_accum 1, timesteps 500` |
| 结构 | 1 层编码 + 3 层解码，4 头，d_state 1；500 步扩散，cosine 调度，L2 损失 | 时域 score 模型：d_model 36，5 层，6 头（适配器默认值，**来历未记录**） | WaveNet：res/skip 128，12 层残差，dilation 循环 6；200 步扩散，β 1e-4→0.02 | d_model 32，1 编码 + 1 解码层，4 头；cosine 调度，L1 损失加 Fourier 损失（厂商默认） |
| 参数量（d1_c1） | **10.51 M**（d1_c2 10.51 M，d7_c1 12.28 M，来自训练日志） | 0.79 M | 2.72 M | 0.22 M |
| 优化器 | Adam，lr 1e-4，betas (0.9, 0.96) | AdamW，lr 峰值 1e-3，前 10% 线性预热，之后余弦衰减 | Adam，lr 2e-4，常数 | Adam，lr 1e-4，betas (0.9, 0.96) |
| 学习率调度 | ⚠️ 配置了 ReduceLROnPlateau 加预热，但厂商 `engine/solver.py:118` 的 `self.sch.step(...)` **被注释掉了**，所以**实际是常数 1e-4** | 预热 + 余弦 | 无 | ReduceLROnPlateau 加预热（升到 8e-4、预热 500 步、因子 0.5、patience 3000、最低 1e-5），**每步都执行** |
| 梯度裁剪 | 1.0 | 1.0 | 无 | 1.0 |
| EMA | 0.995，每 10 步更新一次 | 无 | 无 | 0.995，每 10 步更新一次 |
| 精度 | FP32 | FP32 | FP32 | FP32 |
| 喂入的序列数 | **12.8 M**（100k × 64 × 2） | ≈ 6.4 M（1115 epoch × 5741） | 6.4 M | 6.4 M |
| checkpoint | 第 100k 步终点 | 最后一个 epoch | 最后一次迭代 | 最后一次迭代 |
| 采样器 | 用 EMA 权重，完整 500 步 DDPM 祖先采样 | 反向 VP-SDE，Euler–Maruyama，1000 步 | DDPM 祖先采样，200 步 | 用 EMA 权重，完整 500 步祖先采样 |
| 后处理 | 采样过程中把 x0 截到 [−1, 1] | 逆标准化，不截断 | 无 | 采样过程中把 x0 截到 [−1, 1] |
| 训练种子 | ⚠️ **所有 27 个模型都是 2026**：`dimts.py:115` 读的是 params 里的 seed，不是 job_seed | job_seed | job_seed | job_seed |
| 每个模型的训练耗时（单卡 A100） | 4.60 h（d1_c1）/ 8.04 h（d1_c2）/ 22.9 h（d7_c1），含采样 | 1.17 h 训练 + 0.3 h 采样 | 0.47 h + 0.02 h | 0.52～0.82 h + 0.05～0.09 h |
| 每个格子合计 | 124 / 216 / 619 GPU-h | 41 / 39 GPU-h | 13 / 13 GPU-h | 15 / 24 GPU-h |

上表是 d1 的配置。七天只跑了 DiM-TS，配置相同，只有 `sample_batch` 改成 100。

**比较是怎样设置的（Q14 的后半部分）**

- **预算**：口径是「喂入的序列数相同，约 6.4 M」。DiffWave、Diffusion-TS、FourierDiffusion 和本文模型（25,000 × 256）都是这样（`baseline_base.body.sh`）。**DiM-TS 例外，是 12.8 M**：文档按 batch 64 算，漏掉了写死的梯度累积 2。在 d7_c1 上，本文模型只有 12,000 × 256 ≈ 3.1 M（Q9）。
- **容量没有对齐。** `igfm.py` 文档字符串里说「其它基线校准到 ±3%」，**对表里这些运行不成立**。以本文模型 2.09 M 为 1×：

  | 基线 | 相对容量 |
  |---|---|
  | Diffusion-TS | 0.11× |
  | FourierDiffusion | 0.38× |
  | DiffWave | 1.30× |
  | DiM-TS | 5.0× |

  三个新基线用的是适配器默认值。Diffusion-TS 的 0.22 M 比它作者用过的任何配置都小。一个约 2.0 M 的 Diffusion-TS 配置（照 `solar.yaml` 的 4/4/96）测过（`results/probe/dts_big_d1_c1.json`），**但没有用于主表**。
- **调参**：**每个基线都没有调参**，一次就用。checkpoint 也都取终点。
- **DiM-TS 的终点和最佳里程碑差很多**，Context-FID 对比如下（扫描结果在 `results/matrix/sweep/`）：

  | 格子 | 最佳里程碑 | 终点 |
  |---|---|---|
  | d1_c1 | 0.0590 | 0.0948 |
  | d1_c2 | 0.0992 | 0.1590 |
  | d7_c1 | 0.1306 | 0.3256 |

  质量门槛按终点算，这是全局约定（`configs/experiment.yaml`）。所以「本文模型的 FID 是 DiM-TS 的 2～3 倍好」是**拿它和 DiM-TS 的终点比**；如果拿最佳和最佳比，七天上两者打平。
- **FourierDiffusion 为什么取最后一个 epoch**：早先保留「最佳 epoch」（`keep_best`）的做法，会让唯一共用的 base 变成一张「彩票」，给全部 26 对比较带来偏差（`docs/PITFALLS.md` §21）。改成最后一个 epoch 之后，FID 反而从 0.254 降到 0.100（d1_c1）。注意 `docs/RUNNING_BASELINES.md` 的 trap 3 还写着「keep_best 必须开」，和实际做法矛盾，以 PITFALLS §21 为准。
- **Diffusion-TS 出过的事故**：曾经因为存档目录按 pid 和 `id(self)` 命名，一个 include 模型从上一个模型的权重采样。修复后**整棵树重训过**，现在全部 54 个模型的 fit_seconds 都不少于 1,834 s，干净。

**copy-paste 天花板**（`copy_paste.py`）：把训练窗口原样存下，`sample(n)` 用 job_seed 有放回地抽 n 个**真实窗口**原样放出。K = N 时约 63.2% 的窗口是唯一的。不训练。

## Q15. 「原始 backbone」与消融

**「Backbone, unmodified」是什么**：同一个 IG-FM，三个模块全关，用原版去相关损失。

- 运行树：`results/runs/loo_igfm_d1_c1`、`loo_igfm_d1_c2`，各 27 个模型。
- PBS：`scripts/pbs/dev/igfm_loo.pbs`、`igfm_stock_loo_d1c2_*.pbs`。

**除模块外完全相同**：

| 项目 | 对比 |
|---|---|
| hidden、层数、头数 | 144 / 4+4 / 8，参数量相同（2.09 M） |
| 步数、有效 batch、微批 | 25,000、256、32 |
| 优化器、EMA、裁剪 | 相同 |
| 采样 | 500 步 Euler + EMA |
| K | 相同 |
| 设计和目标 | 相同（`subjects_sha1` 相同） |
| 每个作业的种子 | **相同**（同名作业同种子） |
| 训练耗时 | 每个模型 4.35 小时，本文模型 4.36 小时 |

**唯一的差别**：

1. coord_mask 开 vs 关
2. level pooling 开 vs 关
3. 去相关损失：分组版 vs 全通道版

**同口径结果**：同一次调用里打分，`results/matrix/quality_final/d1_c*.json` 和 `results/matrix/membership_null.json`：

| | d1_c1 FID | d1_c1 暴露 | d1_c2 FID | d1_c2 暴露 |
|---|---|---|---|---|
| backbone | 0.0390 | 1/26 | 0.0546 | 1/26 |
| +2 模块 | 0.0391 | 1/26 | 0.0547 | 3/26 |

**两个轴上都分不出来。** 随机期望是 1.3/26，backbone 本来就在随机水平，没有下降空间。

**没做的消融，以及没验证的结论**：

- **没有单模块消融**：只开改动一、或只开改动二的版本都没跑。所以就算有效果，也分不清是哪个模块的。
- **七天没有原版 backbone**，「This work」在 d7_c1 上没有同口径的对照。
- **低容量 backbone 上的消融没跑**：在随机地板还没被占满的地方，模块才有机会显出作用。`RESULTS.md` §2 标着「未跑，等你定」。
- **结论**：「新增模块有可重复的增益」**不成立，也没有被验证。** 能支持的只有关于架构的陈述：flow-matching 的 backbone 在本数据上处于随机水平，同时质量最好。

---

# 四、复现与结果对应关系

## Q16. 论文结果 → 实验运行

所有路径都相对仓库根目录，**设计**统一是 `results/matrix/design/rep1`（26 对，`subjects_sha1(base)=8cf77b7732b8fe79`）。

**两个评估脚本**

| 指标 | 脚本 | 输出 |
|---|---|---|
| FID | `scripts/eval_quality_tsgem.py`，经 `scripts/pbs/dev/quality_all_cells.pbs`，**每个格子一次调用** | `results/matrix/quality_final/<cell>.json` |
| 暴露 | `scripts/membership_null.py` | `results/matrix/membership_null.json`，键是 `<cell>/<arm>` |

FID 打分设置：只给 **base 的释放**打分，subsample 2,000，seed 2026。

**本文模型和原版 backbone**

| 表中行 / 格子 | 运行树（27 个模型） | 配置 | 代码版本 | checkpoint | FID 用的样本 | 表中数字（FID / 暴露） |
|---|---|---|---|---|---|---|
| This work / d1_c1 | `results/runs/loo_igfm_priv_d1_c1` | `.params.json`；PBS 是 `igfm_priv_loo_{a,b}.pbs` + `igfm_priv_loo.body.sh`；base 来自 `igfm_priv_base.pbs` | `181c0ca` 时的模型代码 | 每个 include 是 `iter-25000.pt`（EMA）；base 的存档在 `results/runs/igfm_priv_d1_c1/base/` | `.../loo_igfm_priv_d1_c1/base/samples.npy` | 0.039 / 1 |
| This work / d1_c2 | `results/runs/loo_igfm_priv_d1_c2` | `.params.json`；PBS 是 `igfm_priv_loo_d1c2_{a,b,c}.pbs`；base 来自 `igfm_priv_base_d1_c2.pbs` | 同上 | 同上；base 存档在 `igfm_priv_d1_c2/base/` | `.../loo_igfm_priv_d1_c2/base/samples.npy` | 0.055 / 3 |
| This work / d7_c1 | `results/runs/loo_igfm_priv_d7_c1` | `.params.json`；PBS 是 `launch_igfm_d7_loo.sh` → `igfm_priv_loo_d7_lane.pbs` + `igfm_priv_loo_cell.body.sh` | `ac13fa2` | include 是 `iter-12000.pt`；**base 摆自** `results/runs/igfm_priv_d7_c1/base/iter-12000.pt` | `.../loo_igfm_priv_d7_c1/base/samples.npy`（摆进来的） | 0.126 / 3 |
| Backbone / d1_c1 | `results/runs/loo_igfm_d1_c1` | `igfm_loo.pbs`（这棵树没有 `.params.json`，参数看各 meta.json） | `34b5482` 的 `igfm_core.py`（上游原样，与现在的 `igfm_core.py.stock` 逐字相同；跑于 2026-09-01，早于模块写入） | `iter-25000.pt` | `.../loo_igfm_d1_c1/base/samples.npy` | 0.039 / 1 |
| Backbone / d1_c2 | `results/runs/loo_igfm_d1_c2` | `.params.json`；`igfm_stock_loo_d1c2_{a,b,c}.pbs` | 同上 | `iter-25000.pt` | `.../loo_igfm_d1_c2/base/samples.npy` | 0.055 / 1 |

**其它臂**：运行树见 `scripts/membership_null.py` 第 53–69 行，配置见 Q13–14。

| 臂 | d1_c1 | d1_c2 | d7_c1 |
|---|---|---|---|
| DiM-TS | `results/runs/matrix_d1_c1` | `matrix_d1_c2` | `matrix_d7_c1` |
| copy-paste | `cp_matrix_d1_c1` | `cp_matrix_d1_c2` | `cp_matrix_d7_c1` |
| DiffWave | `bl_diffwave_d1_c1` | `bl_diffwave_d1_c2` | — |
| FourierDiffusion | `bl_fourier_diff_d1_c1` | `bl_fourier_diff_d1_c2` | — |
| Diffusion-TS | `bl_diffusion_ts_d1_c1` | `bl_diffusion_ts_d1_c2` | — |

**每个运行目录**里都有：`meta.json`（params、job_seed、K、耗时、续训标记）、`samples.npy`（释放集）、最近两个 `iter-*.pt`（摆进来的 base 除外）。训练日志在 `logs/igfm_priv_loo_shard*.log`、`logs/igfm_priv_loo_d7_c1_shard*.log` 等。

## Q17. 运行环境与算力

- **硬件**：NSCC Singapore ASPIRE 2A，项目号 11704243，**NVIDIA A100-SXM4-40GB**（日志里 nvidia-smi 的输出）。**每个模型只用 1 张卡**，多卡作业是几个分片各占一张卡并行。
- **软件**：conda 环境 `/data/projects/11704243/lcheng/envs/cgmoutlier`；torch **2.7.0+cu126**（`constraints.txt` 锁死）、numpy 1.24.4、scipy 1.10.1，驱动 CUDA 12.8。Python 小版本没记录，可以在环境里 `python -V` 查。DiM-TS 另有一个环境 `$WORK/envs/dimts`（cuda 11.8；按建环境脚本是 torch 2.0.1+cu118，但有一处注释写的是 2.12，实际版本没查到）。完整依赖见 `requirements.txt`、`constraints.txt`、`pyproject.toml`。
- **耗时**：来自各 meta.json，**训练时间按单卡计**。

| 运行树 | 模型数 | 每个模型训练（中位数） | 每个模型采样 | 合计训练 | 合计采样 |
|---|---|---|---|---|---|
| This work d1_c1 | 27 | 4.36 h | 0.23 h | 117.8 h | 6.3 h |
| This work d1_c2 | 27 | 4.38 h | 0.23 h | 118.4 h | 6.3 h |
| This work d7_c1（26 个 include） | 26 | **49.8 h**（`fit_seconds_total`，跨续训累加） | 5.4 h | 1,295 h | 146 h |
| d7_c1 base 的源运行，25,000 步 | 1 | 约 104 h 累计（`RESULTS.md` §6；meta 里的 62.4 h 只是续训后最后一段） | 5.4 h | | |
| Backbone d1_c1 | 27 | 4.35 h | 0.23 h | 117.6 h | 6.2 h |
| Backbone d1_c2 | 27 | 4.38 h | 0.23 h | 118.2 h | 6.3 h |

  七天采样比一天慢约 23 倍：T 长 7 倍，注意力是 O(T²)，还要乘上 500 步。
- `fit_seconds_total` 是**有效训练时间，是账单的下界**：被杀掉、没赶上存档的那一截不计入。

## Q18. 公开程度

**这一条需要作者决定，我只列现状。**

- **本项目代码**：仓库 `github.com/Cranooooooo/Metabonet_CGM_MIA`，公开还是私有我没核实。所有 PBS 脚本、适配器、评估脚本都在仓库里，`paper-npjdm-src.zip` 是构建产物（已 gitignore）。
- **IG-FM 上游** `Cranooooooo/IGFM_ICDE2027`：上游 README 写着 "code-only repository: model checkpoints and generated samples are intentionally NOT included"。论文发表前能否公开，取决于那篇 ICDE 稿件的状态。
- **权重**：每个运行目录有 `iter-*.pt`（d1 每个约 37 MB，d1_c1 一棵树 1.8 GB，d7_c1 一棵树 3.0 GB）。**发布权重前请先考虑：在单个受试者上训练出来的 include 模型，本身就是成员推断的对象**，公开它们等于公开 26 个人的成员身份。建议最多只发布 base 模型，或者干脆不发权重。
- **合成样本**：同样的顾虑，include 模型的释放集带有成员信号。
- **数据**：MetaboNet 公开部分，无需 DUA。派生 cohort 可以由 `scripts/build_cohort_multi.py` 加 `build_cohort_matched.py` 重建。数据存放和 DOI 还没开始（`RESULTS.md` §7）。可以用 `nature-data` 技能起草 Data/Code Availability。

---

## 附：写作 AI 要的文件清单 → 仓库里的位置

| 要的东西 | 位置 |
|---|---|
| 模型定义 | `vendor/IG-FM/igfm_core.py`（本文模型），`vendor/IG-FM/igfm_core.py.stock`（原版） |
| registry / adapter | `src/cgmoutlier/generators/registry.py`、`igfm.py`、`base.py` |
| 训练与采样驱动 | `src/cgmoutlier/loo/train.py`、`scripts/run_loo.py` |
| 实际运行配置 | 各树 `.params.json`、各模型 `meta.json`、`scripts/pbs/dev/igfm_priv.params` |
| 训练日志 | `logs/igfm_priv_*.log`、`logs/igfm_loo*.log`、`logs/igfm_priv_loo_d7_c1_shard*.log` |
| checkpoint 选择记录 | 一天：固定取最后一步（无记录可选）；七天：`results/matrix/curve/d7_c1_quality.json` 加 `docs/PAPER_PLAN.md` 第 655–800 行 |
| backbone 与消融 | `results/runs/loo_igfm_d1_c*`；对比见 `quality_final` 和 `membership_null.json` |
| 上游出处 | `vendor/IG-FM/PROVENANCE.md`、`UPSTREAM_README.md` |
| 环境 | `requirements.txt`、`constraints.txt`、`pyproject.toml`、`scripts/pbs/env.sh` |
| 已知坑和历史 | `docs/PITFALLS.md`（§11 预算对齐、§21 共享 base 偏移、§23 跨作业打分漂移、§24 续训耗时） |
