# Brain_2018_kfold_learning

Included is the function used to do nested cross-validation in Vogel et al., 2018 Brain:
https://academic.oup.com/brain/advance-article/doi/10.1093/brain/awy093/4969934

I had only created this as a function for my own use -- had I expected to be publishing it, I would have gone about things completely different. Lesson learned!

But, as a results, its not in the most ideal form to be shared. The code is not modularized and is pretty ugly. I did my best to add some reasonable documentation. I also included a Jupyter notebook with some example usage. However, functionally, its the same code used in the paper, so I guess there's that for reproducibility.

Please note that, in its unfinished state, there are some features that don't work. The function will not support classification problems at this time, and only accepts certain types of sklearn models. While I have the wherewithall to add these features and generally improve the code, I don't have a ton of time. 

However, I'm also assuming little interest. If you have interest in me fixing or adding any features into this function, please don't hesitate to raise an issue and I'll do my best to get right on it. Similarly, if you have interest in any of the other analyses from the paper, raise an issue and let me know.

