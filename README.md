# anly605-app

## Overview

This app was for a group project for my Machine Learning class during the Fall 2022 semester. The goal of this project was to train a model that could distinguish between music tastes, using a year's worth of Spotify streaming history for two of the group members (me and Clare!). 

Our best model was a stacked model with 60% accuracy. We used our model to build a Flask app that allows a user to input any song and artist, and it will return whether the model classifies the song as best fitting Clare's or Nina's data. It will also provide a picture of the album cover, as well as a sample of that song (this data was extracted from the Spotify API). 

## To Run 
1. Download the `app` folder
2. Inside Terminal (or any Python IDE), navigate to inside the `app` folder and install the required dependencies using: 
`pip install -r requirements.txt`
4. Run the app: `flask run`
5. Copy and paste the generated URL into Google Chrome.
6. Have fun! 

