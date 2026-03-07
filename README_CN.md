<p align="center">
  <img src="images/banner.svg" alt="PreMap Banner" width="100%"/>
</p>

<p align="center">
  <strong>面向二倍体李属全同胞家系的端到端基因型处理与清洗软件</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python 3.8+"/></a>
  <a href="#下载"><img src="https://img.shields.io/badge/平台-Linux%20%7C%20Windows-0078D4?logo=linux&logoColor=white" alt="Platform"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/许可证-CC%20BY--NC--ND%204.0-EAB308?logo=creativecommons&logoColor=white" alt="License"/></a>
  <a href="https://doi.org/10.1093/hr/uhaf087"><img src="https://img.shields.io/badge/DOI-10.1093%2Fhr%2Fuhaf087-blue" alt="DOI"/></a>
  <a href="https://github.com/fanjiaqi777/PreMap-V1.0-release/stargazers"><img src="https://img.shields.io/github/stars/fanjiaqi777/PreMap-V1.0-release?style=social" alt="Stars"/></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <b>中文</b>
</p>

---

## 概述

**PreMap** 将原始变异检测结果（VCF 4.2+）或自定义标记矩阵转换为清洗完毕、相位统一、可直接用于图谱构建的基因型 bin 矩阵——全流程自动化，每一步修改均有完整审计追踪。本软件面向二倍体李属（*Prunus*）全同胞家系设计，接受任意符合二等位编码规则的标记类型（SNP、INDEL 等），输出结果可直接用于主流遗传图谱构建软件。

<p align="center">
  <img src="images/Pepline-eng-01.jpg" alt="PreMap 流程总览" width="85%"/>
</p>

### 功能亮点

| 功能 | 说明 |
|:-----|:-----|
| **双通道输入** | 从 VCF 导入 SNP 与 INDEL，或通过 Excel 导入任意二等位标记 |
| **伪测交策略** | 自动分离类型分类（1:1、1:2:1、亲本特异） |
| **稳健相位校正** | s/l/m 三阈值控制，连锁相位矫正与 IBD 检测 |
| **重组断点识别** | 个体水平断点识别 + 群体水平最小重组单位划分 |
| **纠错与填补** | 基因型校正与缺失填补，全过程留痕 |
| **零依赖部署** | 预编译可执行文件，无需安装 Python |
| **跨平台** | Linux 命令行 + Windows 图形界面 |

---

## 下载

<table>
<tr>
<td width="50%" align="center">

### Linux

命令行界面

**[下载 PreMap-linux-v1.0-release.tar.gz](https://github.com/fanjiaqi777/PreMap-V1.0-release/releases/download/v1.0/PreMap-linux-v1.0-release.tar.gz)**

或克隆仓库：
```bash
git clone https://github.com/fanjiaqi777/PreMap-V1.0-release.git
cd PreMap-V1.0-release/dist
chmod +x run_step*
```

无需 Python 环境。

</td>
<td width="50%" align="center">

### Windows

图形用户界面 (GUI)

**[下载 PreMap_v1.0.exe](https://github.com/fanjiaqi777/PreMap-V1.0-release/releases/download/v1.0/PreMap_v1.0.exe)**

双击运行，无需安装。
适用于中小规模数据集。

</td>
</tr>
</table>

---

## 流程模块

```
Step 0 ─── VCF 规范化、染色体拆分与基因型转码 ──────── (仅 Linux)
Step 0.5 ─ 计算基因型频率与统计指标
Step 1 ─── 清洗、过滤与伪测交分离分类
Step 2 ─── 相位校正与聚类清洗（IBD 检测）
Step 3 ─── Binning —— 个体水平重组断点识别
Step 4 ─── OneBins —— 群体水平 bin marker 鉴定、纠错与填补
Step-Add ─ 附加统计（测序数据专用，杂交验证）
```

---

## 快速开始（Linux）

```bash
# Step 0: VCF 处理（仅 Linux）
./dist/run_step0 -i input.vcf

# Step 0.5: Excel 预处理
./dist/run_step0_5 -i input_file.xlsx

# Step 1–4: 主流程
./dist/run_step1 -i chr1_markers.xlsx
./dist/run_step2 -i chr1_cleaned.xlsx
./dist/run_step3 -i chr1_phased.xlsx
./dist/run_step4 -i chr1_binned.xlsx

# 附加步骤：加权校正（测序数据）
./dist/run_step_addition_caculation -i chr1_onebins.xlsx
```

---

## 系统要求

| 规模 | 样本数 | 最低配置 | 推荐配置 |
|:-----|:-------|:---------|:---------|
| 小 | ≤ 150 | 4 核 CPU，8 GB RAM，20 GB 磁盘 | 8 核 CPU，8 GB RAM，SSD |
| 中 | 150–400 | 8 核 CPU，32 GB RAM，100 GB | 16 核 CPU，64 GB RAM，SSD |
| 大 | ≥ 400 | 16 核 CPU，64 GB RAM，200 GB SSD | 24–32 核 CPU，128 GB RAM，NVMe |

> **提示**：Step 0 因数据量大、内存占用高，仅支持 Linux。其余步骤 Linux/Windows 均可运行。

---

## 使用文档

完整的中英双语使用手册（含参数说明、输入模板、截图等）请查看：

**[📖 打开使用手册 (English / 中文)](https://fanjiaqi777.github.io/PreMap-V1.0-release/User-Guide.html)**

> **离线使用**：克隆仓库后，用浏览器打开 `docs/User-Guide.html`。

---

## 测试数据

可复现的测试数据集已发布在 Figshare：

**[https://doi.org/10.6084/m9.figshare.30043073.v2](https://doi.org/10.6084/m9.figshare.30043073.v2)**

---

## 引用

如果在研究中使用了 PreMap，请引用：

> Fan J, Wu J, Arús P, Li Y, Cao K, Wang L (2025). Integrating whole-genome resequencing and machine learning to refine QTL analysis for fruit quality traits in peach. *Horticulture Research*, **12**(7): uhaf087.
> [https://doi.org/10.1093/hr/uhaf087](https://doi.org/10.1093/hr/uhaf087)

---

## 许可证

本项目采用 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) 许可证发布——**学术研究**与**非商业育种活动**可免费使用。禁止商业使用、分发修改版本及反向工程。完整条款详见 [LICENSE](LICENSE)。

如需商业授权或集成授权，请联系 [fjq690510307@gmail.com](mailto:fjq690510307@gmail.com)。

---

## 致谢

- **开发团队**：桃种质资源与育种创新团队
- **共创团队**：IRTA 蔷薇科遗传与基因组研究团队
- **参考的外部工具**：VCF 4.2+ 规范、bcftools、vcftools

---

## 联系方式

**范家琪** — [fjq690510307@gmail.com](mailto:fjq690510307@gmail.com)

GitHub: [@fanjiaqi777](https://github.com/fanjiaqi777)

---

<p align="center">
  <sub>基于 Python · Pandas · NumPy · Jupyter 开发</sub>
</p>
