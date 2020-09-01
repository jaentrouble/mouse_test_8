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
> Sample layer : Conv1D_7 ; last layer of an eye

>![image](https://user-images.githubusercontent.com/45917844/91656397-61623500-eaf3-11ea-9455-5c41b2f30d39.png)

  3. Looks like the agent has trapped in local minima.

- Two things to try : higher epsilon or higher learning rate.
  - In last Bonus test (mouse_test_7), having low learning rate did stop maxQ value from increasing any farther, but maxQ value still shaked a bit.
  - However, in this test, maxQ value had stopped to total flat, so I decided to tweak epsilon first.

### Epsilon 1 to 0.1 for 1.5M steps

>![image](https://user-images.githubusercontent.com/45917844/91663585-c08c6d80-eb24-11ea-8124-84f1230fc36d.png)

>![image](https://user-images.githubusercontent.com/45917844/91663599-d0a44d00-eb24-11ea-9606-f4a8e9f44ca0.png)

>![image](https://user-images.githubusercontent.com/45917844/91663604-da2db500-eb24-11ea-95e4-57d748b8aad5.png)

>![image](https://user-images.githubusercontent.com/45917844/91663632-15c87f00-eb25-11ea-882d-08d29a27e226.png)

1. The exact same thing happened. Suddenly maxQ stopped after 1.7M steps

2. Some sort of a bug?

- Same hyperparameters, without any mixed_precision things(loss scaling was still in my code) and no reloading (as memory leaking is not a big issue right now, with 32G of ram.)

### Without reloading & loss scaling ... 1

>![image](https://user-images.githubusercontent.com/45917844/91784103-02103c00-ec3d-11ea-84de-b654d09a5cab.png)

>![image](https://user-images.githubusercontent.com/45917844/91784113-09374a00-ec3d-11ea-9290-326210de785d.png)

>![image](https://user-images.githubusercontent.com/45917844/91784128-148a7580-ec3d-11ea-9690-25260bb70976.png)

>![image](https://user-images.githubusercontent.com/45917844/91784188-3c79d900-ec3d-11ea-8a88-38bc3a72f8b1.png)

1. Seems like it's working now.

2. See if it was a reloading problem or loss scaling problem

### With reloading, same conditions

>![image](https://user-images.githubusercontent.com/45917844/91784501-0d179c00-ec3e-11ea-956d-0c46ca10bad6.png)

>![image](https://user-images.githubusercontent.com/45917844/91784514-13a61380-ec3e-11ea-83c2-3979e34f92a9.png)

>![image](https://user-images.githubusercontent.com/45917844/91784536-1dc81200-ec3e-11ea-9807-a04aef1d1d24.png)

>![image](https://user-images.githubusercontent.com/45917844/91784569-2d475b00-ec3e-11ea-9cf8-3d67f064a076.png)

1. It happened again.

2. However, this time, it is a little bit different, that there is some small fluctuation in maxQ value.

3. Moreover, to conclude that it's reloading problem, there are still some suspicious things. The point when maxQ value suddenly stops fluctuating is not consistent, even with same conditions. Even more, in this test, reloading was by every 500k steps, but the critical point is in between the reloading points.

4. To see if it is just a matter of luck (or randomness), did one more test without reloading, in a very same condition as the test right before this test.

### Without reloading ... 2

>![image](https://user-images.githubusercontent.com/45917844/91786044-b6ac5c80-ec41-11ea-826a-ebd3095e5317.png)

>![image](https://user-images.githubusercontent.com/45917844/91786059-bd3ad400-ec41-11ea-9aff-13c41414bd88.png)

>![image](https://user-images.githubusercontent.com/45917844/91786065-c330b500-ec41-11ea-89db-82c3708c7562.png)

>![image](https://user-images.githubusercontent.com/45917844/91786088-ce83e080-ec41-11ea-836d-266ebb42dc4c.png)

1. This time, it happened from the beginning.

2. It looks like it's not some kind of a bug. There seems to be a local minima.

3. Considering the result of 'without reloading 1' test, maybe the learning rate is too low

- Try : modify learning rate.

### learning rate 0.01 ~ 0.0005 (decaying over 2M steps)

>![image](https://user-images.githubusercontent.com/45917844/91826800-078b7780-ec79-11ea-9ba7-bd8ef47faeb8.png)

>![image](https://user-images.githubusercontent.com/45917844/91826831-12460c80-ec79-11ea-83ed-8ef1b70331df.png)

>![image](https://user-images.githubusercontent.com/45917844/91826866-1a9e4780-ec79-11ea-8f52-eb1eb168db7d.png)

>![image](https://user-images.githubusercontent.com/45917844/91826909-2853cd00-ec79-11ea-92a0-4edb559209b8.png)

1. Again, there was no update.

2. Checked the eval recordings. Every time when the model stops learning, it was when the model chose only one action for any states. In this environment, there is no punishment, which makes it almost impossible to get out of the local minima when the model gives big Q value to a single action, regardless of states. Maximization bias may be the problem.

- Now it's time to implement some new things to the learning algorithm.


## Discussion

1. Even with the exact same code, the result can vary much.

2. I think this is time to move on to other algorithms. First, try Double DQN and/or Dueling DQN

## TODO

1. Implement Double DQN

2. Implement Dueling DQN

- See any one of them can help learning without changing the environment any farther.
