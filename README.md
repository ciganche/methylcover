
Methylcover is an *in silico* deconvolution tool for inferring relative abundances of 31 cell types from WGBS composite samples. It is based on set-cover-selected region in combination with standard deviation weighted NNLS. 

It supports human genome references hg19 and hg38 - these are set using the ```--genome``` parameter.


### Installation:

```
git clone https://github.com/ciganche/methylcover.git


# test 
cd methylcover
./methylcover beta_deconv -i leukocytes_test_beta.bed -g hg19
```


### Workflow:

1. Use ```methylcover prepare_bam``` to generate input files for beta and read-based deconvolution from a BAM file

2. Use ```methylcover beta_deconv``` or ```methylcover read_level_deconv``` based deconvolution for inferring cell type percentages in a composite mixture. 


### Dependencies:
- python 3+
  * numpy
  * scipy
  * pandas
- [samtools](http://www.htslib.org/) (for ```prepare_bam``` utility)
- [bedtools](https://bedtools.readthedocs.io/en/latest/) (for ```prepare_bam``` utility)
- [wgbstools](https://github.com/nloyfer/wgbs_tools) (for ```prepare_bam``` utility)
