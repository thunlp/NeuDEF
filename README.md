# NeuDEF

This is the reference implementation of the Neural Document Expansion with User Feedback (NeuDEF) model from paper "Neural Document Expansion with User Feedback". 

## Requirements

- python 2.7
- torch 0.4.1
- numpy 1.16.3

## Efficiency

During training, it takes about 300ms to process one batch on a single-GPU machine with the following settings:

- batch size: 64
- max_q_len: 10
- max_d_len: 50
- max_body_len: 100
- max_exp_len: 10
- max_exp_num: 10
- vocabulary_size: 200K
- embedding dimension: 300
- multi-head attention layer: 1
- multi-head attention head: 4
- learning rate: 0.001

## Results

Please refer our [paper](https://arxiv.org/pdf/1908.02938.pdf).

## Citation

arXiv version:

```
@article{yin2019neural,
  title={Neural Document Expansion with User Feedback},
  author={Yin, Yue and Xiong, Chenyan and Luo, Cheng and Liu, Zhiyuan},
  journal={arXiv preprint arXiv:1908.02938},
  year={2019}
}
```



## Contact

If you have questions, suggestions and bug reports, please email bnuyinyue@outlook.com.
