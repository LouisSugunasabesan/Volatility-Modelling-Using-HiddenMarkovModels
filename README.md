# Volatility-Modelling-Using-HIddenMarkovModels
Applying Hidden Markov Models to model Gold Intraday Volatility by detecting regime switches from low-volatility regimes to high-volatility

- Hidden Markov Models (HMMs) are a class of probabilistic graphical model that allow us to predict a sequence of unknown (hidden) variables from a set of observed variables. A simple example of an HMM is predicting the weather (hidden variable) based on the type of clothes that someone wears (observed)." 

https://medium.com/@postsanjay/hidden-markov-models-simplified-c3f58728caab#:~:text=Hidden%20Markov%20Models%20(HMMs)%20are,that%20someone%20wears%20(observed).

The aim of this repository is to share with you my foray into volatility modelling using HiddenMarkovModels (HMM's.) HMM's were picked for this for their state-ful nature. Markets have been shown to exist in regimes, states if you were, and if you can predict when a market ia bout to switch regime - sya from low volatility to high volatility it can be monetized. This is what I have done within this repo.

I have modelled the volatility regime intraday (1D Candles) of Gold Prices (obtained from NASDAQ) from 2013-2021. The model aims to provide you with a graph which alternates from 0 (High Volatility) to 1 (Low Volatility) providing direction from when the Gold market is about to switch regime.

I have chosen to make this piece of research public as intra-day investing does not suit my personal style and I am eternally grateful to the entire open source community and so felt it was only right for me to give back.

All data and notebooks are provided.
