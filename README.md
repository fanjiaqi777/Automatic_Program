<p align="center">
  <img src="images/banner.svg" alt="PreMap Banner" width="100%"/>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python 3.8+"/></a>
  <a href="#-download"><img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-0078D4?logo=windowsterminal&logoColor=white" alt="Platform"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-EAB308?logo=creativecommons&logoColor=white" alt="License"/></a>
  <a href="https://doi.org/10.1093/hr/uhaf087"><img src="https://img.shields.io/badge/DOI-10.1093%2Fhr%2Fuhaf087-1674B1?logo=doi&logoColor=white" alt="DOI"/></a>
  <a href="https://github.com/fanjiaqi777/PreMap-V1.0-release/releases"><img src="https://img.shields.io/github/v/release/fanjiaqi777/PreMap-V1.0-release?color=00C853&logo=github" alt="Release"/></a>
  <a href="https://github.com/fanjiaqi777/PreMap-V1.0-release/stargazers"><img src="https://img.shields.io/github/stars/fanjiaqi777/PreMap-V1.0-release?style=social" alt="Stars"/></a>
</p>

<p align="center">
  <b>English</b> | <a href="README_CN.md">中文</a>
</p>

<br/>

> **PreMap** transforms raw variant calls (VCF 4.2+) or custom marker matrices into clean, phased, map-ready genotype bins — fully automated, with a complete audit trail. Designed for diploid *Prunus* full-sib families, it accepts any biallelic marker type (SNP, INDEL, etc.) and outputs directly compatible with mainstream linkage-mapping software.

<br/>

## 🧬 Pipeline

<p align="center">
  <img src="images/Pepline-eng-01.jpg" alt="PreMap Pipeline" width="90%"/>
</p>

<details>
<summary><b>Module details</b></summary>
<br/>

| Module | Function | Platform |
|:-------|:---------|:--------:|
| **Step 0** | VCF normalization, chromosome splitting & genotype recoding | Linux |
| **Step 0.5** | Genotype frequency & statistics computation | All |
| **Step 1** | Cleaning, filtering & pseudo-testcross segregation classification | All |
| **Step 2** | Phase correction, cluster cleaning & IBD detection | All |
| **Step 3** | Individual-level recombination breakpoint identification (binning) | All |
| **Step 4** | Population-level OneBins — bin-marker correction & imputation | All |
| **Step-Addition** | Additional statistics for hybrid verification (sequencing data) | All |

</details>

<br/>

## ⚡ Key Features

<table>
<tr>
<td width="50%">

**🧪 Biallelic Marker Support**
SNP, INDEL, and any custom biallelic markers from VCF or Excel input.

**🔗 Robust Phasing**
Triple-threshold (s/l/m) phase correction with automatic IBD detection.

**📊 Recombination Binning**
Individual-level breakpoint identification & population-level minimal recombination units.

</td>
<td width="50%">

**🔍 Full Traceability**
Complete audit trail — every correction and imputation is logged.

**📦 Zero Dependencies**
Pre-built executables. No Python, no pip, no setup.

**🖥️ Cross-Platform**
Linux command-line + Windows GUI — same pipeline, your choice.

</td>
</tr>
</table>

<br/>

## 📥 Download

<table>
<tr>
<td width="50%" align="center">
<br/>
<img src="https://img.shields.io/badge/Linux-CLI-FCC624?logo=linux&logoColor=black&style=for-the-badge" alt="Linux"/>
<br/><br/>

**[⬇ PreMap-linux-v1.0-release.tar.gz](https://github.com/fanjiaqi777/PreMap-V1.0-release/releases/download/v1.0/PreMap-linux-v1.0-release.tar.gz)**

Or clone:
```bash
git clone https://github.com/fanjiaqi777/PreMap-V1.0-release.git
```
Executables in `dist/` — no Python needed.
<br/><br/>
</td>
<td width="50%" align="center">
<br/>
<img src="https://img.shields.io/badge/Windows-GUI-0078D6?logo=windows&logoColor=white&style=for-the-badge" alt="Windows"/>
<br/><br/>

**[⬇ PreMap_v1.0.exe](https://github.com/fanjiaqi777/PreMap-V1.0-release/releases/download/v1.0/PreMap_v1.0.exe)**

Double-click to run. No installation required.
For small to medium-scale datasets.
<br/><br/>
</td>
</tr>
</table>

<br/>

## 🖥️ Windows GUI Preview

PreMap provides a complete graphical interface on Windows — configure parameters, browse files, and run each step with one click:

<p align="center">
  <img src="images/windows1.png" alt="Step 1: Clean and Filter" width="70%"/>
</p>

<details>
<summary><b>More screenshots</b></summary>
<br/>
<p align="center">
  <img src="images/windows3.png" alt="Step 3: Bins Selection" width="70%"/>
</p>
<p align="center">
  <img src="images/windows4.png" alt="Step 4: OneBins Processor" width="70%"/>
</p>
</details>

<br/>

## 🚀 Quick Start (Linux)

```bash
# Step 0: VCF processing (Linux only)
./dist/run_step0 -i input.vcf

# Step 0.5: Compute statistics
./dist/run_step0_5 -i input_file.xlsx

# Step 1–4: Main workflow
./dist/run_step1 -i chr1_markers.xlsx
./dist/run_step2 -i chr1_cleaned.xlsx
./dist/run_step3 -i chr1_phased.xlsx
./dist/run_step4 -i chr1_binned.xlsx

# Additional: Weighted correction (sequencing data)
./dist/run_step_addition_caculation -i chr1_onebins.xlsx
```

<br/>

## 💻 System Requirements

| Scale | Samples | Minimum | Recommended |
|:------|:--------|:--------|:------------|
| Small | ≤ 150 | 4-core, 8 GB RAM, 20 GB disk | 8-core, 8 GB RAM, SSD |
| Medium | 150–400 | 8-core, 32 GB RAM, 100 GB | 16-core, 64 GB RAM, SSD |
| Large | ≥ 400 | 16-core, 64 GB RAM, 200 GB SSD | 24–32-core, 128 GB RAM, NVMe |

> **Note**: Step 0 is Linux-only (large data volume). All other steps run on both Linux and Windows.

<br/>

## 📖 Documentation

Full bilingual User Guide with step-by-step instructions, parameter references, and input templates:

**[📖 Open User Guide (English / 中文)](https://htmlpreview.github.io/?https://github.com/fanjiaqi777/PreMap-V1.0-release/blob/main/docs/User-Guide.html)**

> **Offline**: Clone the repo and open `docs/User-Guide.html` in your browser.

<br/>

## 🧪 Test Data

A reproducible test dataset is available on Figshare:

[![DOI](https://img.shields.io/badge/Figshare-10.6084%2Fm9.figshare.30043073.v2-blue?logo=figshare&logoColor=white)](https://doi.org/10.6084/m9.figshare.30043073.v2)

<br/>

## 📝 Citation

If you use PreMap in your research, please cite:

> Fan J, Wu J, Arús P, Li Y, Cao K, Wang L (2025). Integrating whole-genome resequencing and machine learning to refine QTL analysis for fruit quality traits in peach. *Horticulture Research*, **12**(7): uhaf087.
> [https://doi.org/10.1093/hr/uhaf087](https://doi.org/10.1093/hr/uhaf087)

<br/>

## 📄 License

This project is licensed under **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)** — free for academic research and non-commercial breeding activities. See [LICENSE](LICENSE) for full terms.

For commercial or integration licenses → [fjq690510307@gmail.com](mailto:fjq690510307@gmail.com)

<br/>

## 🙏 Acknowledgments

- **Development team** — Peach Germplasm Resources and Breeding Innovation Team
- **Collaborating team** — IRTA Rosaceae Genetics and Genomics Research Team
- **External references** — VCF 4.2+ spec, bcftools, vcftools

<br/>

---

<p align="center">
  <b>Jiaqi Fan</b> &nbsp;·&nbsp; <a href="mailto:fjq690510307@gmail.com">fjq690510307@gmail.com</a> &nbsp;·&nbsp; <a href="https://github.com/fanjiaqi777">@fanjiaqi777</a>
</p>

<p align="center">
  <sub>Built with Python · Pandas · NumPy · Jupyter</sub>
</p>
