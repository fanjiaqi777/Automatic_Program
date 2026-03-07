<p align="center">
  <img src="images/banner.svg" alt="PreMap Banner" width="100%"/>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python 3.8+"/></a>
  <a href="#-下载"><img src="https://img.shields.io/badge/平台-Linux%20%7C%20Windows-0078D4?logo=windowsterminal&logoColor=white" alt="Platform"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/许可证-CC%20BY--NC--ND%204.0-EAB308?logo=creativecommons&logoColor=white" alt="License"/></a>
  <a href="https://doi.org/10.1093/hr/uhaf087"><img src="https://img.shields.io/badge/DOI-10.1093%2Fhr%2Fuhaf087-1674B1?logo=doi&logoColor=white" alt="DOI"/></a>
  <a href="https://github.com/fanjiaqi777/PreMap-V1.0-release/releases"><img src="https://img.shields.io/github/v/release/fanjiaqi777/PreMap-V1.0-release?color=00C853&logo=github" alt="Release"/></a>
  <a href="https://github.com/fanjiaqi777/PreMap-V1.0-release/stargazers"><img src="https://img.shields.io/github/stars/fanjiaqi777/PreMap-V1.0-release?style=social" alt="Stars"/></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <b>中文</b>
</p>

<br/>

> **PreMap** 将原始变异检测结果（VCF 4.2+）或自定义标记矩阵转换为清洗完毕、相位统一、可直接用于图谱构建的基因型 bin 矩阵——全流程自动化，每一步修改均有完整审计追踪。面向二倍体李属（*Prunus*）全同胞家系设计，接受任意符合二等位编码规则的标记类型（SNP、INDEL 等），输出可直接用于主流遗传图谱构建软件。

<br/>

## 🧬 流程概览

<p align="center">
  <img src="images/Pepline-eng-01.jpg" alt="PreMap 流程总览" width="90%"/>
</p>

<details>
<summary><b>模块详情</b></summary>
<br/>

| 模块 | 功能 | 平台 |
|:-----|:-----|:----:|
| **Step 0** | VCF 规范化、染色体拆分与基因型转码 | Linux |
| **Step 0.5** | 基因型频率与统计指标计算 | 全平台 |
| **Step 1** | 清洗、过滤与伪测交分离分类 | 全平台 |
| **Step 2** | 相位校正、聚类清洗与 IBD 检测 | 全平台 |
| **Step 3** | 个体水平重组断点识别（Binning） | 全平台 |
| **Step 4** | 群体水平 OneBins — bin marker 纠错与填补 | 全平台 |
| **Step-Addition** | 附加统计，杂交验证（测序数据专用） | 全平台 |

</details>

<br/>

## ⚡ 功能亮点

<table>
<tr>
<td width="50%">

**🧪 二等位标记支持**
从 VCF 或 Excel 导入 SNP、INDEL 及任意自定义二等位标记。

**🔗 稳健相位校正**
s/l/m 三阈值控制，自动 IBD 检测与连锁相位矫正。

**📊 重组断点识别**
个体水平断点识别 + 群体水平最小重组单位划分。

</td>
<td width="50%">

**🔍 全程可追溯**
完整审计追踪——每一次纠错与填补均有记录。

**📦 零依赖部署**
预编译可执行文件，无需 Python，无需 pip，无需配置。

**🖥️ 跨平台**
Linux 命令行 + Windows 图形界面——同一流程，自由选择。

</td>
</tr>
</table>

<br/>

## 📥 下载

<table>
<tr>
<td width="50%" align="center">
<br/>
<img src="https://img.shields.io/badge/Linux-CLI-FCC624?logo=linux&logoColor=black&style=for-the-badge" alt="Linux"/>
<br/><br/>

**[⬇ PreMap-linux-v1.0-release.tar.gz](https://github.com/fanjiaqi777/PreMap-V1.0-release/releases/download/v1.0/PreMap-linux-v1.0-release.tar.gz)**

或克隆仓库：
```bash
git clone https://github.com/fanjiaqi777/PreMap-V1.0-release.git
```
可执行文件位于 `dist/`，无需 Python 环境。
<br/><br/>
</td>
<td width="50%" align="center">
<br/>
<img src="https://img.shields.io/badge/Windows-GUI-0078D6?logo=windows&logoColor=white&style=for-the-badge" alt="Windows"/>
<br/><br/>

**[⬇ PreMap_v1.0.exe](https://github.com/fanjiaqi777/PreMap-V1.0-release/releases/download/v1.0/PreMap_v1.0.exe)**

双击运行，无需安装。
适用于中小规模数据集。
<br/><br/>
</td>
</tr>
</table>

<br/>

## 🖥️ Windows 图形界面预览

PreMap 在 Windows 上提供完整的图形界面——配置参数、选择文件、一键运行：

<p align="center">
  <img src="images/windows1.png" alt="Step 1: 清洗与过滤" width="70%"/>
</p>

<details>
<summary><b>更多截图</b></summary>
<br/>
<p align="center">
  <img src="images/windows3.png" alt="Step 3: Bins 识别" width="70%"/>
</p>
<p align="center">
  <img src="images/windows4.png" alt="Step 4: OneBins 处理器" width="70%"/>
</p>
</details>

<br/>

## 🚀 快速开始（Linux）

```bash
# Step 0: VCF 处理（仅 Linux）
./dist/run_step0 -i input.vcf

# Step 0.5: 统计指标计算
./dist/run_step0_5 -i input_file.xlsx

# Step 1–4: 主流程
./dist/run_step1 -i chr1_markers.xlsx
./dist/run_step2 -i chr1_cleaned.xlsx
./dist/run_step3 -i chr1_phased.xlsx
./dist/run_step4 -i chr1_binned.xlsx

# 附加步骤：加权校正（测序数据）
./dist/run_step_addition_caculation -i chr1_onebins.xlsx
```

<br/>

## 💻 系统要求

| 规模 | 样本数 | 最低配置 | 推荐配置 |
|:-----|:-------|:---------|:---------|
| 小 | ≤ 150 | 4 核 CPU，8 GB RAM，20 GB 磁盘 | 8 核 CPU，8 GB RAM，SSD |
| 中 | 150–400 | 8 核 CPU，32 GB RAM，100 GB | 16 核 CPU，64 GB RAM，SSD |
| 大 | ≥ 400 | 16 核 CPU，64 GB RAM，200 GB SSD | 24–32 核 CPU，128 GB RAM，NVMe |

> **提示**：Step 0 因数据量大，仅支持 Linux。其余步骤 Linux/Windows 均可运行。

<br/>

## 📖 使用文档

完整的中英双语使用手册（含参数说明、输入模板、截图等）：

**[📖 打开使用手册 (English / 中文)](https://htmlpreview.github.io/?https://github.com/fanjiaqi777/PreMap-V1.0-release/blob/main/docs/User-Guide.html)**

> **离线使用**：克隆仓库后，用浏览器打开 `docs/User-Guide.html`。

<br/>

## 🧪 测试数据

可复现的测试数据集已发布在 Figshare：

[![DOI](https://img.shields.io/badge/Figshare-10.6084%2Fm9.figshare.30043073.v2-blue?logo=figshare&logoColor=white)](https://doi.org/10.6084/m9.figshare.30043073.v2)

<br/>

## 📝 引用

如果在研究中使用了 PreMap，请引用：

> Fan J, Wu J, Arús P, Li Y, Cao K, Wang L (2025). Integrating whole-genome resequencing and machine learning to refine QTL analysis for fruit quality traits in peach. *Horticulture Research*, **12**(7): uhaf087.
> [https://doi.org/10.1093/hr/uhaf087](https://doi.org/10.1093/hr/uhaf087)

<br/>

## 📄 许可证

本项目采用 **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)** 许可证发布——学术研究与非商业育种活动可免费使用。完整条款详见 [LICENSE](LICENSE)。

如需商业授权或集成授权 → [fjq690510307@gmail.com](mailto:fjq690510307@gmail.com)

<br/>

## 🙏 致谢

- **开发团队** — 桃种质资源与育种创新团队
- **共创团队** — IRTA 蔷薇科遗传与基因组研究团队
- **参考工具** — VCF 4.2+ 规范、bcftools、vcftools

<br/>

---

<p align="center">
  <b>范家琪</b> &nbsp;·&nbsp; <a href="mailto:fjq690510307@gmail.com">fjq690510307@gmail.com</a> &nbsp;·&nbsp; <a href="https://github.com/fanjiaqi777">@fanjiaqi777</a>
</p>

<p align="center">
  <sub>基于 Python · Pandas · NumPy · Jupyter 开发</sub>
</p>
