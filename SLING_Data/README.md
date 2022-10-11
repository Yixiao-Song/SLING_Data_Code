This folder contains the SLING dataset accompanying the paper "SLING: Sino LINGuistic Evaluation of Large Language Models".

Each phenomenon is a separate folder in this directory, with paradigms being single text files in the phenomenon folder.

Every line in the paradigm file represents one minimal pair, separated by @@@ symbol. The acceptable sentence always comes before the unacceptable one.

```
<good-sentence>@@@<bad-sentence>

他们上个月到底勾结了什么?@@@什么他们上个月到底勾结了?
你们前天究竟转换成了什么?@@@什么你们前天究竟转换成了?
她们上个月究竟引爆了什么?@@@什么她们上个月究竟引爆了?
他们上个月到底加快了什么?@@@什么他们上个月到底加快了?
```

Note that in the `MP_Classifier/mp_cl_comp_noun_v2.txt` and `MP_Classifier/mp_cl_adj_comp_noun_v2` files, the first 98 pairs are manually generated and the rest 902 pairs are code generated.
