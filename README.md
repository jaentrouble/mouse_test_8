# mouse_test_8

 More apples

## Purpose

Increase apple number And see if that helps.

## Lessons from last experiment

  1. Stop using mixed precision. The model is not computationaly heavy, so using mixed precision has no benefit here.

  2. Current algorithm is able to learn cart-pole environment. Not only does it reach maximum reward, it holds the maximum reward continuously.
    - This means it's either environment's problem or hyperparameter problem.

  3. Bonus 1 test shows that learning rate do effect learning significantly. Changing learning rate will affect learning.

  4. Using prioritized replay did not help learning in current environment.

## Last time TODO

  1. Change sparsity of the reward. It can be either reducing the map size or increasing the number of apples.

## Plan

  1. Increase the number of apples.

## Other changes

  1. Deleted mouse - apple distance computation.

## Tests

### 10 Apples

>![image](https://user-images.githubusercontent.com/45917844/91656258-7094b300-eaf2-11ea-9811-a3cc160583a0.png)

>![image](https://user-images.githubusercontent.com/45917844/91656264-78545780-eaf2-11ea-9716-f4022004a2ab.png)

>![image](https://user-images.githubusercontent.com/45917844/91656273-80ac9280-eaf2-11ea-8d15-4ec5669956f2.png)
  
  1. As seen in last test (mouse_test_7), after about 1.4M steps, maxQ suddenly stopped changing.
> Test2

>![image](https://user-images.githubusercontent.com/45917844/91637635-d165b200-ea44-11ea-8cb8-4972a56a39c4.png)

> Test3

>![image](https://user-images.githubusercontent.com/45917844/91637689-34efdf80-ea45-11ea-98e9-e1df8e394b67.png)

  2. Histogram shows there's no update to the weights after 1.4M steps.

  3. Looks like the agent has trapped in local minima.

- Two things to try : higher epsilon or higher learning rate.
  - In last Bonus test (mouse_test_7), having low learning rate did stop maxQ value from increasing any farther, but maxQ value still shaked a bit.
  - However, in this test, maxQ value had stopped to total flat, so I decided to tweak epsilon first.

## Discussion

## TODO
