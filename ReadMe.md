**Deep Code Search using RNN**

The paper helps search for code snippets described by the user. In a huge codebase of more than million lines of code, manually searching for it will not be easy. The existing methods use text based approach which are not as good. This approach uses deep learning, which helps encode the semnatics of code in a better way. A bi-directional LSTM is used, which predicts the future as well as previous tokens of a sentence. Therefore, with deep learning this paper could achieve much better results however there are inaccuracies since the deep learning models still don't know much about the programming context while training.

**How to Run?**

* Search

  `python sde.py`
* Train

  For training, in configs.py, set `reload=-1` and in sde.py set mode as `train` and run the following:

  `python sde.py`
