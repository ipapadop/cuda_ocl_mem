# cuda_ocl_mem
Simple example on how to access host mapped memory from CUDA and OpenCL. Intended as an example on CUDA/OpenCL interoperability.

Before using `CL_MEM_USE_HOST_PTR` please read the following about some caveats:
- https://forums.khronos.org/showthread.php/6184-Creating-buffers
- https://stackoverflow.com/questions/17775738/cl-mem-alloc-host-ptr-slower-than-cl-mem-use-host-ptr
- https://www.nvidia.com/content/cudazone/CUDABrowser/downloads/papers/NVIDIA_OpenCL_BestPracticesGuide.pdf
- https://software.intel.com/en-us/node/540453
- http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0538f/CIHEGGGA.html
