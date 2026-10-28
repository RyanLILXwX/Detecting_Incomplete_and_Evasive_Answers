# Comparison of Four Qwen3 Full-Parameter Fine-Tuning Results

This table compares the best checkpoint selected from each of the four training runs. Values are rounded to six decimal places, and classification metrics range from 0 to 1. Bold indicates the best value in each row; loss and sample counts are not highlighted.

The existing ranking rule is used: compare `macro_f1` first, then `accuracy` if tied, with higher values preferred for both. Rankings use the original precision in the JSON files. The final model version is chosen by validation-set ranking; rankings on the two test sets are shown only to compare performance, not to select checkpoints.

| Dataset | Metric | Qwen3-1.7B whole_qa | Qwen3-1.7B request | Qwen3-4B whole_qa | Qwen3-4B request |
|---|---|---:|---:|---:|---:|
| validation | Rank (existing ranking rule) | 2 | **1** | 4 | 3 |
| validation | accuracy | 0.716148 | **0.733701** | 0.712136 | 0.721163 |
| validation | macro_f1 | 0.653888 | **0.670917** | 0.629198 | 0.650549 |
| validation | weighted_f1 | 0.718105 | **0.733700** | 0.708857 | 0.720549 |
| validation | direct.precision | **0.824903** | 0.803109 | 0.777593 | 0.794737 |
| validation | direct.recall | 0.728522 | 0.798969 | **0.804983** | 0.778351 |
| validation | direct.f1 | 0.773723 | **0.801034** | 0.791051 | 0.786458 |
| validation | direct.support | 1164 | 1164 | 1164 | 1164 |
| validation | intermediate.precision | 0.590551 | **0.647966** | 0.613409 | 0.620915 |
| validation | intermediate.recall | **0.752149** | 0.661891 | 0.616046 | 0.680516 |
| validation | intermediate.f1 | **0.661626** | 0.654855 | 0.614725 | 0.649351 |
| validation | intermediate.support | 698 | 698 | 698 | 698 |
| validation | fully_evasive.precision | **0.714286** | 0.577236 | 0.602273 | 0.640449 |
| validation | fully_evasive.recall | 0.416667 | **0.537879** | 0.401515 | 0.431818 |
| validation | fully_evasive.f1 | 0.526316 | **0.556863** | 0.481818 | 0.515837 |
| validation | fully_evasive.support | 132 | 132 | 132 | 132 |
| validation | sample_count | 1994 | 1994 | 1994 | 1994 |
| validation | validation_loss | 0.759229 | 0.770738 | 0.781543 | 1.273919 |
| Training interval corresponding to the selected checkpoint | training_loss | 0.414600 | 0.398496 | 0.287655 | 0.163980 |
| test | Rank (same metric priority) | 4 | **1** | 2 | 3 |
| test | accuracy | 0.694995 | 0.716714 | 0.719547 | **0.727101** |
| test | macro_f1 | 0.622111 | **0.654019** | 0.645532 | 0.643027 |
| test | weighted_f1 | 0.695873 | 0.717548 | 0.716650 | **0.725403** |
| test | direct.precision | **0.818708** | 0.798142 | 0.776722 | 0.804837 |
| test | direct.recall | 0.695902 | 0.774590 | **0.804098** | 0.790984 |
| test | direct.f1 | 0.752326 | 0.786190 | 0.790173 | **0.797850** |
| test | direct.support | 1220 | 1220 | 1220 | 1220 |
| test | intermediate.precision | 0.565776 | 0.622028 | **0.638184** | 0.627427 |
| test | intermediate.recall | **0.764706** | 0.664439 | 0.639037 | 0.691176 |
| test | intermediate.f1 | 0.650370 | 0.642534 | 0.638611 | **0.657761** |
| test | intermediate.support | 748 | 748 | 748 | 748 |
| test | fully_evasive.precision | **0.728571** | 0.562963 | 0.613208 | 0.610526 |
| test | fully_evasive.recall | 0.340000 | **0.506667** | 0.433333 | 0.386667 |
| test | fully_evasive.f1 | 0.463636 | **0.533333** | 0.507813 | 0.473469 |
| test | fully_evasive.support | 150 | 150 | 150 | 150 |
| test | sample_count | 2118 | 2118 | 2118 | 2118 |
| human_test | Rank (same metric priority) | 4 | **1** | 3 | 2 |
| human_test | accuracy | 0.668342 | **0.715243** | 0.693467 | 0.698492 |
| human_test | macro_f1 | 0.641166 | **0.690970** | 0.661904 | 0.663497 |
| human_test | weighted_f1 | 0.670195 | **0.715401** | 0.688192 | 0.697457 |
| human_test | direct.precision | 0.761246 | **0.780564** | 0.733894 | 0.768997 |
| human_test | direct.recall | 0.685358 | 0.775701 | **0.816199** | 0.788162 |
| human_test | direct.f1 | 0.721311 | 0.778125 | 0.772861 | **0.778462** |
| human_test | direct.support | 321 | 321 | 321 | 321 |
| human_test | intermediate.precision | 0.549091 | **0.642857** | 0.610526 | 0.587444 |
| human_test | intermediate.recall | **0.708920** | 0.633803 | 0.544601 | 0.615023 |
| human_test | intermediate.f1 | 0.618852 | **0.638298** | 0.575682 | 0.600917 |
| human_test | intermediate.support | 213 | 213 | 213 | 213 |
| human_test | fully_evasive.precision | **0.848485** | 0.632353 | 0.720000 | 0.733333 |
| human_test | fully_evasive.recall | 0.444444 | **0.682540** | 0.571429 | 0.523810 |
| human_test | fully_evasive.f1 | 0.583333 | **0.656489** | 0.637168 | 0.611111 |
| human_test | fully_evasive.support | 63 | 63 | 63 | 63 |
| human_test | sample_count | 597 | 597 | 597 | 597 |

**Conclusion: Under the existing ranking rule, Qwen3-1.7B + request is the best version in this comparison.** The validation-set ranking is: **1.7B request > 1.7B whole_qa > 4B request > 4B whole_qa**. The winning model has a validation Macro-F1 of **0.670917** and also ranks first on both test sets with Macro-F1 scores of **0.654019** and **0.690970**, respectively.

It does not lead on every metric: for example, **4B request** has the highest Accuracy and Weighted-F1 on `test`. These conclusions are limited to the four runs with a single seed (1101) and do not establish a statistically significant advantage.

Metric definitions: Classification metrics and `sample_count` are computed at the original QA level for both modes, and `support` is the number of ground-truth samples in each class. Loss is computed per sub-question for `request` and per complete QA for `whole_qa`, so it cannot be compared directly across tasks. `training_loss` is the average training loss over the validation interval immediately preceding the selected checkpoint, not over the entire training run; the test result files do not record loss.