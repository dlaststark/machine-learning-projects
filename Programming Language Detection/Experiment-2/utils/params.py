#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to define the reusable parameters

Created on Mon Nov  2 14:38:25 2020

@author: tapasdas
"""


base_proj_path = '/content/drive/My Drive/Colab Notebooks/Programming Language Detection/Experiment-2'
class_wght_fl = base_proj_path + '/Saved Files/class_weights.txt'
embed_models_fl = base_proj_path + '/Saved Files/sentence_embed_models.txt'
npz_fl = base_proj_path + '/Saved Files/Prog_Lang_Dataset.npz'
scaler_fl = base_proj_path + '/Saved Files/data_scaler.txt'
artifacts_path = base_proj_path + '/Artifacts'
predictions_path = base_proj_path + '/Predictions'

lang_map = {
     'Ada': 0
    ,'AWK': 1
    ,'C': 2
    ,'C++': 3
    ,'Clojure': 4
    ,'D': 5
    ,'Erlang': 6
    ,'Fortran': 7
    ,'Go': 8
    ,'Groovy': 9
    ,'Haskell': 10
    ,'J': 11
    ,'Java': 12
    ,'JavaScript': 13
    ,'Julia': 14
    ,'Lua': 15
    ,'Mathematica': 16
    ,'MATLAB': 17
    ,'Perl': 18
    ,'PHP': 19
    ,'PowerShell': 20
    ,'Python': 21
    ,'R': 22
    ,'REXX': 23
    ,'Ruby': 24
    ,'Rust': 25
    ,'Scala': 26
    ,'Smalltalk': 27
    ,'Swift': 28
    ,'UNIX-Shell': 29
    ,'Tcl': 30
    ,'Zkl': 31
    ,'Jq': 32
    ,'Racket': 33
    ,'Kotlin': 34
}

rev_lang_map = {
     0: 'Ada'
    ,1: 'AWK'
    ,2: 'C'
    ,3: 'C++'
    ,4: 'Clojure'
    ,5: 'D'
    ,6: 'Erlang'
    ,7: 'Fortran'
    ,8: 'Go'
    ,9: 'Groovy'
    ,10: 'Haskell'
    ,11: 'J'
    ,12: 'Java'
    ,13: 'JavaScript'
    ,14: 'Julia'
    ,15: 'Lua'
    ,16: 'Mathematica'
    ,17: 'MATLAB'
    ,18: 'Perl'
    ,19: 'PHP'
    ,20: 'PowerShell'
    ,21: 'Python'
    ,22: 'R'
    ,23: 'REXX'
    ,24: 'Ruby'
    ,25: 'Rust'
    ,26: 'Scala'
    ,27: 'Smalltalk'
    ,28: 'Swift'
    ,29: 'UNIX-Shell'
    ,30: 'Tcl'
    ,31: 'Zkl'
    ,32: 'Jq'
    ,33: 'Racket'
    ,34: 'Kotlin'
}

