# keras_starter

This is a keras starter project that provides tf.summary logging out of the box 
inside your model inference code.

You should be able to see something like

![Screenshot](images/screenshot.png)

where 
- eval has exactly 3 records (at 0, 10 and 20 corresponding to 3 `evaluate`s)
- train has 20 records (0-9 and 10-19 corresponding to 2 `fit`s with 10 epochs each)

 
