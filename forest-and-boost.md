# Prompts: Random Forest & Gradient Boosting experiments

Verbatim record of the prompts from the random forest / gradient boosting
portion of the chat session that produced the `bcancer-random-forest` and
`bcancer-gradient-boosting` MLflow experiments in `bcancer.py`.

## 1

yeah, i think i would like to try randomforestclassifier, as another experiment, does it have hyperparameters, that can be tweaked and ran in multiple experiments, make sure to preserve all the existing experiments, don't run the bcancer.py, ill run it when you have it ready

## 2

logreg still wins, can we tweak a few more params and see if we can overfit a forest, just kidding

## 3

how about gradient boost experiements, and tell me why we don't combine all experiments together, is it because hyperparameters are diff per model

## 4

one more push back, why is logreg combined with decision trees, still, just because baseline?

## 5

yeah lets clean it up that waqy

## 6

errors below, wait, am i better off letting you run it, so i never see these errors, or is this a better learing technique, dont be nice, let me hear the truth

```
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x0000018C740FCFE0>
Traceback (most recent call last):
  File "C:\Users\tsteele\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\Lib\tkinter\__init__.py", line 414, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: main thread is not in main loop
Tcl_AsyncDelete: async handler deleted by the wrong thread
```
