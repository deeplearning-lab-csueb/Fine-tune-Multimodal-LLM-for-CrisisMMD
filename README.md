# Fine-tune-Multimodal-LLM-for-CrisisMMD

## CrisisMMD dataset
Original CrisisMMD dataset: https://crisisnlp.qcri.org/crisismmd 

We use the same benchmark splits introduced by Ofli *et al.* (2020), and only consider tweets where the text and image share the same label for consistency. Additionally, we followed the practice in Mandal *et al.* (2024) and merged a few semantically similar classes to streamline classification. Specifically, *“injured or dead people”* and *“missing or found people”* were consolidated into *“affected individuals”*, while *“vehicle damage”* was grouped under *“infrastructure and utility damage”*.

The class distributions across the train/dev/test splits of the dataset for the Tweet Informativeness and Humanitarian Category tasks are as follows:
#### Class distribution for *Informativeness* and *Humanitarian Category* tasks
|                 |                      | ** Text** |          |            |                | **Image ** |           |          |          |
|----------------|-----------------------|----------------|--------------|---------------|----------------|------------------|---------------|----------------|------------------|
| **Task**       | **Category**          | ** Train** | ** Dev** | ** Test** | ** Total** | ** Train** | ** Dev** | ** Test** | ** Total** |
|----------------|-----------------------|----------------|--------------|---------------|----------------|------------------|---------------|----------------|------------------|
| **Informative** | Informative           | 5,546          | 1,056        | 1,030         | 7,632          | 6,345            | 1,056         | 1,030          | 8,431            |
|                | Not-informative       | 2,747          | 517          | 504           | 3,768          | 3,256            | 517           | 504            | 4,277            |
|                | **Total**             | 8,293          | 1,573        | 1,534         | 11,400         | 9,601            | 1,573         | 1,534          | 12,708           |
| **Humanitarian**| Affected individuals  | 70             | 9            | 9             | 88             | 71               | 9             | 9              | 89               |
|               | Rescue/Volunteering   | 762            | 149          | 126           | 1,037          | 912              | 149           | 126            | 1,187            |
|               | Infrastructure damage | 496            | 80           | 81            | 657            | 612              | 80            | 81             | 773              |
|              | Other relevant        | 1,192          | 239          | 235           | 1,666          | 1,279            | 239           | 235            | 1,753            |
|               | Not-humanitarian      | 2,743          | 521          | 504           | 3,768          | 3,252            | 521           | 504            | 4,277            |
| | **Total**             | 5,263          | 998          | 955           | 7,216          | 6,126            | 998           | 955            | 8,079            |



## Zero-shot, One-shot and partial Five-shot experiments
Zero-shot, One-shot experiments of GPT-4o, GPT-4o mini, Llama 3.2 11B models are at Anh-New/Performance_Comparison_Improved_July2025.ipynb
Five-shot experiments on Humanitarian classification task with GPT-4o, GPT-4o mini Anh-New/Performance_Comparison_Improved_July2025.ipynb

## Fine-tuning LLaMA 3.2 11B with LoRA
Code in LLaMA/LLaMA-3.2-11B
### Hyper-parameters for Informative and Humanitarian Tasks (LLaMA 3.2 11B)

| Hyper-parameter | Informative - Text only | Informative - Image only | Informative - Text + Image | Humanitarian - Text only | Humanitarian - Image only | Humanitarian - Text + Image |
|-----------------|-------------------------|---------------------------|-----------------------------|---------------------------|-----------------------------|-------------------------------|
| rank            | 16                      | 8                         | 16                          | 24                        | 8                           | 24                            |
| batch size      | 16                      | 1                         | 16                          | 16                        | 1                           | 16                            |
| learning rate   | 5e-4                    | 1e-4                      | 1e-3                        | 5e-4                      | 1e-4                        | 1e-3                          |
| epoch           | 3                       | 3                         | 2                           | 3                         | 3                           | 3                             |
