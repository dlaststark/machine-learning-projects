mat=: 0 ". ];._2 noun define
 1  0  0  0  0  0  1  0  0  0  0 _1  0  0  0  0  0  0
 1  0  0  0  0  0  0  1  0  0  0  0 _1  0  0  0  0  0
 1  0  0  0  0  0  0  0  1  0  0  0  0 _1  0  0  0  0
 0  1  0  0  0  0  1  0  0  0  0  0  0  0 _1  0  0  0
 0  1  0  0  0  0  0  0  1  0  0 _1  0  0  0  0  0  0
 0  1  0  0  0  0  0  0  0  0  1  0  0  0  0  0 _1  0
 0  0  1  0  0  0  1  0  0  0  0  0 _1  0  0  0  0  0
 0  0  1  0  0  0  0  0  0  1  0  0  0  0 _1  0  0  0
 0  0  0  1  0  0  0  1  0  0  0  0  0  0  0 _1  0  0
 0  0  0  1  0  0  0  0  0  1  0  0 _1  0  0  0  0  0
 0  0  0  0  1  0  0  1  0  0  0  0  0 _1  0  0  0  0
 0  0  0  0  1  0  0  0  1  0  0  0  0  0  0  0 _1  0
 0  0  0  0  1  0  0  0  0  0  1  0  0  0  0 _1  0  0
 0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  1
 0  0  0  0  0  1  0  0  0  0  1  0  0  0 _1  0  0  0
)
      gauss_jordan mat
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.435897
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.307692
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.512821
0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0.717949
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0.487179
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0        0
0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0.205128
0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0.282051
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0.333333
0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0        0
0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0.512821
0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0.641026
0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0.717949
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0.769231
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0.512821
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0        1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0.820513
