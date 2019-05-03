# bag-of-words

Code for the bag-of-words model

Preprocessed data files not updated because they're huge and well above GitHub's file size limit of 100 MB. 

If you want to generate the BOW data yourself, you can just run `python preprocessed.py`, but first I would recommend generating a random subset of the data. This can be done by modifying the line `simplified = df.sample(frac=1.0)` in  `preprocessed.py`. Changing the `frac` value specifies the random fraction of the data you would like to use. 
