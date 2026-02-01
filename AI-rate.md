Scaling language models to billions of parameters has unleashed emergent capabilities in program

ming, mathematical reasoning, and multi-step problem solving. GPT-4 (OpenAI [2024]) and Gemini

Ultra (Gemini Team [2025]) demonstrate expert-level performance on competitive programming 

benchmarks and graduate-level science exams, while smaller models like Llama-3-70B (Llama Team 

[2024]) achieve strong results on tool use and long-context reasoning. However, the growth of model 

parameters brings prohibitive memory footprints, latency, and energy costs, making deployment on 

edge devices or task-specific specialization difficult.

Parameter-efficient fine-tuning (PEFT) has been proposed as an efficient alternative to fully finetuning 

(FFT) for specializing LLMs on downstream tasks. Instead of updating all parameters, PEFT methods 

inject small trainable modules while freezing the pretrained backbone. Houlsby et al. [2019] proposes 

adapter layers with bottleneck architectures, achieving near full fine-tuning performance on natural 

language understanding tasks. Prefix-tuning (Li and Liang [2021]) prepends learnable tokens to 

keys and values, reducing trainable parameters to 0.1% of original models. LoRA (Hu et al. [2022]) 

enables task adaptation with less than 0.01% trainable parameters and no inference latency by adding 

decomposed low-rank matrices to linear layers. The successes of LoRA have motivated several 

follow-up enhancements. AdaLoRA (Zhang et al. [2023]) dynamically allocates parameter budgets

across layers according to the importance scores of the weights. RS-LoRA (Kalajdzievski [2023]) 

introduces a rank stabilization scaling factor to address slow learning and performance bottleneck 

with higher ranks. OLoRA (Büyükakyüz [2024]) improves the LLMs training convergence with 

orthogonal initialization of low-rank matrices. DoRA (Liu et al. [2024]) decomposes pretrained 

weights into magnitude and direction components and applied LoRA to the direction component to 

minic the learning strategy of FFT, enhancing learning capacity without extra parameters. These 

LoRA variants have improved training stability, learning capacity, and convergence speed of LoRA. 

However, the fundamental expressiveness bottleneck of a single rank decomposition has not been 

addressed.

