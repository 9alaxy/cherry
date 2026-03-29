## Plan: 峰值显存四方法对比补全

目标是在不改变你当前实验核心（3模型×5数据集×4方法，主指标仅峰值显存）的前提下，把实验从“设计描述”补齐为“可批量执行、可复现、可作图、可追溯日志”的方案。优先先打通 Betty 方法与统一方法定义，再补全批量脚本与统计口径，最后固化出图规范。
当前临时范围调整：目前暂时取消对 ogbn-products 数据集的实验，待后续恢复后再补跑。


**Steps**
0. Phase 0 - 冒烟测试（smoke 集合）
0.1 每种方法至少选取1个小数据集（如 karate/cora），用最小配置（1 epoch、1 run）先跑通，验证：
	- 脚本能正常启动和结束
	- 日志（训练日志、nvidia-smi 采样日志）完整产出
	- 峰值显存字段可被后续汇总脚本识别
	- OOM/超时能被正确记录且不打断批任务

1. Phase A - 方法定义与范围冻结
1.1 在实验文档中补充“方法ID对照表”，明确4种方法固定为 Berry、Betty、DGL_random、DGL_metis，并给出每种方法在脚本参数层面的唯一映射。  
1.2 明确范围边界：本轮只产出峰值显存主图（4张，临时不含 ogbn-products）；精度和时间仅作为可选附录日志，不纳入主结论。  
1.3 明确硬件与运行约束：仅 RTX 3090，默认 GPU 设备号为 1；每次运行必须同时产出训练日志与 nvidia-smi 采样日志。  
1.4 固定目录规范：该实验相关脚本统一放在 /workspace/Cherry/Evaluation/berry/peak_mem_exp；日志统一放在 /workspace/Cherry/Evaluation/berry/peak_mem_exp/log。  

2. Phase B - Betty 可执行性闭环（depends on 1）
2.1 盘点 Betty 现有代码入口，确认训练入口、参数接口、日志输出是否与另外3方法统一。  
2.2 若 Betty 训练实现缺失或不完整，先定义最小可比较版本（同模型结构、同 fan-out、同 batch 数、同 epoch 数、同种子策略）。  
2.3 对齐 Betty 的峰值显存采集口径：框架侧峰值（torch.cuda.max_memory_allocated）与系统侧峰值（nvidia-smi memory.used）同时记录。  
2.4 为所有方法与模型增加失败可追溯机制：OOM/超时单独记录，不中断全局 sweep。  

3. Phase C - 参数标准化与公平性控制（parallel with 2.2/2.3）
3.1 固定公共参数模板：num-batch=8、层数与 hidden 按模型固定、fan-out 按数据集固定、优化器/学习率/dropout 固定。  
3.2 部分参数组合可能导致 OOM，这是正常的现象，如实记录即可。  
3.3 增加预热与采样窗口规则：跳过首轮初始化抖动（例如首 epoch 仅预热不计入峰值统计，或明确仅统计稳定阶段）。  
3.4 统一“峰值显存”定义：以每次运行内全流程最大值为准，并明确单位统一为 GB。  

4. Phase D - 全矩阵批量执行与日志规范（depends on 1-3）
4.1 新增总控脚本，自动遍历 3模型×4数据集×4方法 共 48 组配置（临时不含 ogbn-products）；支持断点续跑。  
4.2 每次实验写入统一命名日志文件：包含方法、模型、数据集、seed、时间戳、设备号。  
4.3 并行记录 GPU 采样日志（nvidia-smi 周期采样）并与训练日志同名关联。  
4.4 失败任务写入 fail manifest（失败原因、退出码、是否 OOM、重试状态）。  

5. Phase E - 汇总与可视化规范（depends on 4）
5.1 生成标准化结果表：dataset、model、method、seed、peak_mem_gb_framework、peak_mem_gb_nvsmi、status。  
5.2 汇总脚本输出每个数据集一张分组柱状图：X 轴方法（4组），每组3个柱（GCN/GAT/SAGE），Y 轴 Peak Memory (GB)。  
5.3 图形规范固定：颜色映射、误差条（IQR 或 std）、坐标范围策略、图例顺序、标题含 GPU 型号与显存总量。  
5.4 产出实验附录：记录未完成/失败配置比例与原因，避免选择性报告。  

6. Phase F - 全量质量门禁与复核（depends on 5）
6.1 做全量前检查：方法覆盖率=100%、配置数=48×runs（临时不含 ogbn-products）、日志双轨完整率=100%。  
6.2 做结果一致性复核：framework 峰值与 nvidia-smi 峰值数量级一致，异常点回溯原始日志。  

**Relevant files**
- /workspace/Cherry/Evaluation/berry/峰值显存对比实验.md — 补充方法定义、统计口径、作图规范、失败策略
- /workspace/Cherry/Evaluation/micro_batch_train_berry.py — 四方法训练入口、峰值显存采集与异常处理对齐
- /workspace/Cherry/Evaluation/Betty.py — Betty 方法训练逻辑与参数兼容性检查
- /workspace/Cherry/Evaluation/Betty_collection.py — Betty 结果采集字段与汇总接口
- /workspace/Cherry/Evaluation/max_memory_collection.py — 峰值显存汇总口径与输出字段标准化
- /workspace/Cherry/Evaluation/berry/run_berry_train.sh — 单任务运行模板（设备号、日志导出）
- /workspace/Cherry/AGENTS.md — GPU=1 与日志双导出约束基线

**Verification**
1. 运行 smoke 矩阵（每方法至少1组）并确认每次运行都有训练日志与 nvidia-smi 日志。
2. 检查汇总 CSV 是否覆盖 4方法×3模型×4数据集×runs 的所有组合（临时不含 ogbn-products），缺失项必须在 fail manifest 可追溯。
3. 随机抽查至少3组配置，对比原始日志与汇总表中的峰值显存是否一致。
4. 生成4张最终图并人工核对每图均包含4种方法与3个模型柱子（临时不含 ogbn-products）。

**Decisions**
- 已确认严格保留文档中的四方法（包含 Betty），不将其替换为 Cherry。
- 主指标仅峰值显存；精度与时间不作为主图结论。
- 纳入强制运行约束：RTX 3090（device 1）+ 双日志导出（训练日志与 GPU 采样日志）。
- 目前暂时取消对 ogbn-products 数据集的实验，后续恢复时单独补跑并回填汇总与图表。

**Further Considerations**
1. Betty 若短期无法补齐训练实现，可采用“阶段性结果”策略：先完成其余3方法全量，再补 Betty 并统一重跑受影响图表。
2. 对 ogbn-papers100M 建议预设更保守超时与重试策略，减少长任务阻塞。
3. 若需要论文级稳健性，可在附录增加“框架峰值 vs nvidia-smi 峰值差异分布图”。
