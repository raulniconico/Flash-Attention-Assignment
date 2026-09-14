**In 6.report.ipynb** provides a detailed account of the tasks listed in Assignment Section 2.2, Task Requirements, of the assignment. 
Although the five questions are answered as five separate tasks in this notebook, they are closely related and build on one another logically.
Given the length constraints, this notebook includes as much of the supporting argument as possible while answering each question with empirical data.
A great deal of the detailed code, testing, and reasoning cannot be shown here in full; all of it is preserved in the five notebooks listed below.

 **1 1.model_profile.ipynb**: A theoretical roofline analysis of Llama 3.1 8B under the given conditions

**2 2.hardware_profile.ipynb**: Discusses in detail the hardware modeling process and the performance analysis, and most importantly, provides a complete codebase for performance analysis.

**3 Flash attention.ipynb**: The design of flash attention, how to optimize the kernel, the choice of tiling settings, and the SRAM allocation strategy

**4 MLP.ipynb**: GEMM algorithm design, focusing on the Q/K/V computation in attention and the parallel computation of the MLP

**5 performance parallelization.ipynb**: Performance analysis using the performance model, together with a discussion of parallelization strategies


For easier reading, PDF versions of the six notebooks can be found in the pdf folder. The environment required to run the notebooks can be installed via install.sh.

Finally, all the code used in the notebooks is located in the code folder.