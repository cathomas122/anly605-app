from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pickle
import uuid
from matplotlib import rcParams
import seaborn as sns
from sklearn.model_selection import train_test_split

import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy

# -------------------------------------------------------------------------- #
#
# SET UP ACCESS TO SPOTIFY API
# 
# -------------------------------------------------------------------------- #
spotify_url = 'https://api.spotify.com/v1'

client_id = '8e910b9aa35e4098ac6dab1c9b9c03e0'

client_secret = '593e14cf88dd479fbf32496e5ae01d9a'

redirect_uri = 'https://www.google.com/'

scope = 'user-library-read'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id = client_id,
                                            client_secret = client_secret,
                                            redirect_uri = redirect_uri,
                                            scope=scope))

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])


def hello_world():
    
    request_type_str = request.method 
    path = "static/confusion_matrix_stacked.jpg" # keep this one static for now

    if request_type_str == 'GET':
        return render_template('index.html', href = path) # this will represent the model's accuracy
    else:
        # -------------------------------------------------------------------- # 
        # LOAD THE DATA
        # -------------------------------------------------------------------- # 
        filename = 'merged_streaming_history.csv' # may need to change the file path 
        df = pd.read_csv(filename, header=0)

        grouped_file = 'merged_grouped.csv'
        grouped_songs = pd.read_csv(grouped_file, header=0) # this data is saved inside the app folder

        # -------------------------------------------------------------------- # 
        # REMOVE UNNECESSARY COLUMNS
        # -------------------------------------------------------------------- # 
        toremove = ["end_time", "artist_name", "track_name", "song_ids", "images", "preview"]
        df = df.drop(toremove, axis=1)

        # -------------------------------------------------------------------- # 
        # CHANGING DATE TO DATETIME & GETTING YEAR ONLY
        # -------------------------------------------------------------------- # 
        df['release_date'] = pd.to_datetime(df['release_date'])
        grouped_songs['release_date'] = pd.to_datetime(grouped_songs['release_date'])

        df['label'] = df['label'].map({'clare': 1, 'nina': 0})
        grouped_songs['label'] = grouped_songs['label'].map({'clare': 1, 'nina': 0})

        df['release_date'] = pd.DatetimeIndex(df['release_date']).year
        grouped_songs['release_date'] = pd.DatetimeIndex(grouped_songs['release_date']).year

        # -------------------------------------------------------------------- #
        # LOAD THE MODEL WITH DETAILS
        # -------------------------------------------------------------------- #
        pkl_filename = "TrainedModel/StackedPickle.pkl"
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)

        # -------------------------------------------------------------------- #
        #
        # NOW WORKING WITH THE INPUTS
        #
        # -------------------------------------------------------------------- #
        text = request.form['text']
        song_image = ""
        song_preview = "www.google.com"


        if text != "":
            print(text)

            # random_string = uuid.uuid4().hex
            # path = "static/" + random_string + ".svg" # this will be the new image that is saved, based on user input
            np_arr = floatsome_to_np_array(text).reshape(1, -1) # reshape makes it all one line
        else:
            print('Array of features was not used. Try a different type of input: ')

            # -------------------------------------------------------------------- #
            #
            # CREATE ANOTHER ARRAY BASED ON SPOTIFY API QUERIES
            #
            # -------------------------------------------------------------------- #

            # -------------------------------------------------------------------- #
            # USER ENTERS THE SONG AND ARTIST
            # -------------------------------------------------------------------- #
            song = request.form['song']
            artist = request.form['artist']
            
            # -------------------------------------------------------------------- #
            # CHECK IF THE SONG NAME & ARTIST ARE IN THE DATASET ALREADY 
            #   – From there, extract the feature values from the dataset, as well
            #     as who listened to it, the number of times they played it
            #   – The more meta data will be output in a separate part of the index.html
            # -------------------------------------------------------------------- #
            print(song)
            print(artist)

           
            avg_freq = sum(df['frequency'])/len(df.index)
            avg_ms = sum(df['ms_played'])/len(df.index) # find average ms_played so that it doesn't have a huge impact on algorithm

            artists = grouped_songs[grouped_songs['track_name'] == song]
            correct = artists[artists['artist_name'] == artist].reset_index(drop = True) # will have multiple rows if the song exists in the dataset

            try:
                if len(correct.index) != 0: # extract features of selected song 
                    feature_values = correct.iloc[0][['mode', 'time_signature', 'key', 'release_date', 'popularity', 'frequency', 'danceability',
                                            'valence', 'tempo', 'speechiness', 'liveness', 'instrumentalness', 'loudness', 'energy',
                                            'acousticness']]
                    feature_values['ms_played'] = avg_ms   

                    print(correct)

                    song_image = correct.iloc[0]['images']
                    song_preview = correct.iloc[0]['preview']

                    print(song_image)
                    print(song_preview)

                    np_arr = floatsome_to_np_array(feature_values.values).reshape(1, -1) # The feature values that are extracted will be assigned to np_arr
                else: # in the event that the song doesn't exist in the dataset
                    print('Searching spotify API')
                    track_id = find_song_id(song, artist)
                    song_features = get_track_features(track_id)
                    
                    all_features_df = pd.DataFrame(columns = ['release_date','popularity','images','preview',
                                                                            'acousticness',
                                                                            'danceability',
                                                                            'energy',
                                                                            'key',
                                                                            'instrumentalness',
                                                                            'liveness',
                                                                            'loudness',
                                                                            'mode',
                                                                            'speechiness',
                                                                            'valence',
                                                                            'tempo',
                                                                            'time_signature'])
                    all_features_df.loc[0] = song_features

                    print(all_features_df)

                    feature_values = all_features_df[['mode', 'time_signature', 'key', 'release_date', 'popularity', 'danceability',
                                            'valence', 'tempo', 'speechiness', 'liveness', 'instrumentalness', 'loudness', 'energy',
                                            'acousticness']]

                    feature_values['release_date'] = pd.to_datetime(feature_values['release_date'])
                    feature_values['release_date'] = pd.DatetimeIndex(feature_values['release_date']).year

                    feature_values['frequency'] = avg_freq
                    feature_values['ms_played'] = avg_ms 

                    song_image = all_features_df.loc[0,'images']
                    song_preview = all_features_df.loc[0, 'preview']
                    
                    np_arr = floatsome_to_np_array(feature_values.iloc[0].values).reshape(1, -1)
            except:
                print('Exception thrown!')
                model_result = 'Song search did not work. Try a different input!'
        try:
            print('Printing np_array')
            print(np_arr) # does not work for the new Spotify API data
            model_result = find_results(pickle_model, np_arr)
        except:
            model_result = 'Model result did not work. Try a different input!'

        return render_template('index.html', href = path, model_result = model_result, 
                                song_image = song_image, song_preview = song_preview)
    

def find_song_id(song, artist):
    not_found = np.nan # set the default return value

    try:
        # returns list of max 10 songs 
        songs = sp.search(q = 'track:' + song, 
                                type='track',
                                limit = 10)
        
        # searches for the artist
        artists = sp.search(q='artist:' + artist, 
                            type='artist') 
        
        # to find the artist id:
        artist_id = artists['artists']['items'][0]['id'] 
        
        # verify the search
        print('\nYou are looking for a song named', song, 'by', artist)
        
        # loop through song search results and compare artist id for each song 
        for i in songs['tracks']['items']:
            song_artist_id = i['artists'][0]['id'] # find artist id
            song_artist_name = i['artists'][0]['name'] # find artist name
        
            # if the artist id from the user input matches the artist if from one of the searched songs
            if song_artist_id == artist_id and i['name'] == song: 
                found_song_name = i['name'] # make sure the found song name matches the user input
                found_artist_name = song_artist_name
                print('You searched for', song, 'by', artist)
                print('and you found', found_song_name, 'by', found_artist_name)
                not_found = i['id'] 
                break
    except: 
        print(song, "produced an error. Returning NaN and moving on!")
        
    return not_found

def get_track_features(track_id):
    meta = sp.track(track_id) # meta data (having trouble calling)
    features = sp.audio_features(track_id) # all track features
    
    release_date = meta['album']['release_date'] # release date
    popularity = meta['popularity'] # song popularity
    images = meta['album']['images'][0]['url'] # album image
    preview = meta['preview_url'] # song preview
    
    # include all features
    try:
        acousticness = features[0]['acousticness']
        danceability = features[0]['danceability'] # 
        energy = features[0]['energy'] #
        key = features[0]['key'] #  -1 = no key detected, 0 = C, 1 = C#/Dflat etc.
        instrumentalness = features[0]['instrumentalness']
        liveness = features[0]['liveness']
        loudness = features[0]['loudness'] # -60 to 0 decibels
        mode = features[0]['mode'] # 1 major, 0 minor
        speechiness = features[0]['speechiness'] 
        valence = features[0]['valence'] # higher valence is more positive
        tempo = features[0]['tempo'] # beats per minute
        time_signature = features[0]['time_signature']
        
        track = [release_date,
                popularity,
                images,
                preview,
                acousticness,
                danceability,
                energy,
                key,
                instrumentalness,
                liveness,
                loudness,
                mode,
                speechiness,
                valence,
                tempo,
                time_signature]
        return track
    except:
        print(track_id, 'did not load properly.')
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def find_results(model, new_input_arr):
    """
    # -------------------------------------------------------------------- # 
    # LOAD THE DATA
    # -------------------------------------------------------------------- # 
    filename = '../merged_streaming_history.csv' # may need to change the file path 
    df = pd.read_csv(filename, header=0)

    # -------------------------------------------------------------------- # 
    # REMOVE UNNECESSARY COLUMNS
    # -------------------------------------------------------------------- # 
    toremove = ["end_time", "artist_name", "track_name", "song_ids", "images", "preview"]
    df = df.drop(toremove, axis=1)

    # -------------------------------------------------------------------- # 
    # CHANGING DATE TO DATETIME & GETTING YEAR ONLY
    # -------------------------------------------------------------------- # 
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['label'] = df['label'].map({'clare': 1, 'nina': 0})
    df['release_date'] = pd.DatetimeIndex(df['release_date']).year
    """

    # -------------------------------------------------------------------- # 
    # FEED NEW INPUT INTO MODEL TO SEE WHAT THE RESULT IS 
    # -------------------------------------------------------------------- # 
    predict = model.predict(new_input_arr)
    predict_str = str(predict)

    # may need to retrain
    if predict_str == '[1]':
        predict_str = 'The combined set of features best fits Clare\'s music taste.'
    else:
        predict_str = 'The combination of features best fits Nina\'s music taste.'
    
    return predict_str

def floatsome_to_np_array(floats_vect):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    try:
        floats = np.array([float(x) for x in floats_vect.split(',') if is_float(x)])
    except:
        floats = np.array([float(x) for x in floats_vect if is_float(x)])
    return floats.reshape(len(floats), 1)
    
