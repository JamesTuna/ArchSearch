# ArchSearch
## DNN architecture search by Gaussian Process

- Train new networks selected with command  
  ```python3.6 PNASsearch/search.py --filename=[inputFile] --start=[startIndex] --end=[endIndex] --index=[GPUIndex] --outfile=[outputFile]```    
  
- Work under folder ```/GradientForGP``` to update parameters in kernel function
  - Implemented by maximizing marginal likelihood of observed samples
  - Normalized first 640 samples in train_gp.txt
  - dist_mat.txt cashed, this is pickle file for distances calculted by graph kernel
  
- ```all_kernel``` much faster for calculating pairwise distance between all the graphs, ```WL_kernel``` between single pair deprecated  
- One way to ensure full rank of conariance matrix: fixed ```sq_n``` to be some value large enough that makes it positive definite (like 0.44)  
- Use reweightForWL to train and assign weights for each dimension of vectors computed by WL kernel. Experiment show that directly using WL kernel is no good for Gaussian process. Here are two graphs illustrating the point.   
## Without weights, difference of performance against Euclidean distance 
  ![not reweighted](https://github.com/JamesTuna/ArchSearch/blob/master/simpleWL.png)   
## After reweighting, difference of performance against Euclidean distance
  ![reweighted](https://github.com/JamesTuna/ArchSearch/blob/master/reweight.png)
