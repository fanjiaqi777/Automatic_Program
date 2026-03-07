# Changelog

All notable changes to PreMap will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0] - 2025-07-01

### Added
- **Step 0**: VCF normalization, chromosome splitting, and genotype recoding (Linux only)
- **Step 0 INDEL**: Dedicated INDEL marker processing pipeline
- **Step 0.5**: Genotype frequency and statistics computation for segregating loci
- **Step 1**: Data cleaning, filtering, and pseudo-testcross segregation classification
- **Step 2**: Phase correction, cluster cleaning, and IBD detection
- **Step 3**: Individual-level recombination breakpoint identification (binning)
- **Step 4**: Population-level OneBins processor with genotype correction and imputation
- **Step-Addition**: Additional statistics for sequencing data (hybrid verification)
- Pre-built executables for Linux (CLI) and Windows (GUI) — zero dependencies
- Chip genotyping test dataset in `test/Chip-data/`
- Comprehensive bilingual User Guide (English / 中文)
