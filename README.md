## WARNING 

The code written is not supposed to be used as any sort of signal for trading in a live market. As a matter of fact, it shouldn't be used  at all. I wrote this code as a fun first project to help me understand the utility of not only the frameworks used, such as scikit-learn, pandas, and numpy, but also the usage of AI.

# THE PROJECT

## Why I Used Each Technology

I will now provide a breakdown of the frameworks and technologies used, why they were used, and when applicable, why I didn't use other technologies and frameworks.

### Python

I used Python as the only programming language this project. While I can see the implications of utilizing something like C++ as it would be faster for predicting stock prices, or even Rust to create something faster that is memory safe, I went with Python because it is the one that I understand best and so debugging was very easy. It also had the libraries I wanted to use, and I was somewhat familiar with pandas and matplotlib in Python, so it gave me all the more incentive to use Python. I want to admit that, if used in a real trading environment, it would not be ideal to use Python over C/C++ when looking at raw performance. I also want to admit that something like Prolog might have been better for the development of an AI system, however I decided to work with Python for my first AI programming project.

### pandas

The gold standard for data analysis and data manipulation in Python, I went ahead with pandas because not only have I used it in previous projects, but also because I will be working with csv files that need cleaning before they are run through the AI Model. Are there alternatives I could have used? Perhaps, but I don't think any would match the quality of pandas, at least in my opinion.

### scikit-learn

I used scikit-learn for my AI model because it was, at least compared to other models, relatively easy to understand and work with. Especially for the LRR model I used (more on this in its respective section), I didn't feel the need to use more advanced frameworks such as Pytorch, Tensorflow, and Keras since scikit-learn provided me with the necessary tools.

### Linear Regression Model

I used a linear regression model to examine the relationship between the 10 day SMA, the 30 day SMA, and the closing price to predict what the closing price was going to be. I'm sure that perhaps something more advanced like a Neural Network would find a better way to examine stock prices, but I didn't want to use anything too advanced for my first AI project, so I stuck with an LRR.

### matplotlib

I wanted to see my results displayed on a screen to see the performance of my model compared to the actual stock price on that given day. This was done with matplotlib.
