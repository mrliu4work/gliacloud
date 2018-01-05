# GliaCloud

## Part 1: Math & Common Sense

### Q1-1: Back Propagation

### Q1-2: Common Sense

-   F1-score 同時考慮了 recall 以及 precision，precision 只考慮了預測的結果是否正確，recall 則考慮了正確結果的數量比例。
-   用 binary classification function 當 activate function 訓練過程當中會很容易導致 gradient vanish
-   variance 表示的是一個 distribution 當中所有 sample 彼此之間的距離，variance 小表示 sample 彼此之間距離短，反之則長。bias 表示的是一個 distribution 與 target distribution 之間的距離，bias 小表示兩個 distribution 距離近，反之則遠。
-   Random Forest 每顆 tree 用的 feature 在訓練之前已經 bagging 過了，如果發生 overfitting 即使做了 pruning 也無法改善。
-   one-hot encoding 是將 categorical value 轉換成一組 of binary value 的方法，舉例
    categorical value => ['Monday', 'Monday', 'Friday', 'Sunday', 'Friday']
    經過 one-hot encoding 會轉換成 [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
-   避免 overfitting:
    1.  do cross validation
    2.  add regularization cost in object function
    3.  add dropout layer
    4.  de-noise 
    5.  go deeper, not wider

## Part 2