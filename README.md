
Methylcover is an *in silico* deconvolution tool for inferring relative abundances of 31 cell types from WGBS composite samples. It is based on set-cover-selected region in combination with standard deviation weighted NNLS. 



Usage:

```
git clone https://github.com/ciganche/methylcover.git
cd methylcover
methylcover <functionality> -h 
```


Workflow:

1. Use ```methylcover prepare_bam``` to generate input files for beta and read-based deconvolution from a BAM file

2. Use ```methylcover beta_deconv``` or ```methylcover read_level_deconv``` based deconvolution for inferring cell type percentages in a composite mixture. 


