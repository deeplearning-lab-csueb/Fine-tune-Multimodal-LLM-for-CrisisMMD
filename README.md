# Fine-tune-Multimodal-LLM-for-CrisisMMD


### Hyper-parameters

### Hyper-parameters for Informative and Humanitarian Tasks (LLaMA 3.2 11B)

| Hyper-parameter | Informative - Text only | Informative - Image only | Informative - Text + Image | Humanitarian - Text only | Humanitarian - Image only | Humanitarian - Text + Image |
|-----------------|-------------------------|---------------------------|-----------------------------|---------------------------|-----------------------------|-------------------------------|
| rank            | 16                      | 8                         | 16                          | 24                        | 8                           | 24                            |
| batch size      | 16                      | 1                         | 16                          | 16                        | 1                           | 16                            |
| learning rate   | 5e-4                    | 1e-4                      | 1e-3                        | 5e-4                      | 1e-4                        | 1e-3                          |
| epoch           | 3                       | 3                         | 2                           | 3                         | 3                           | 3                             |
