import pandas as pd
from moviepy.editor import *
from get_dataframe import get_dataframe 


def generate_highlights(match_video):

    end_time1 = []
    end_time2 = []

    start_time1 = []
    start_time2 = []

    data_f = get_dataframe(match_video)
    df = pd.read_csv(data_f)
    clip = VideoFileClip(match_video)


    df["Score1"] = df["Score1"].replace(["O"], "0")
    df["Score2"] = df["Score2"].replace(["O"], "0")


    df["Score1_chnage"] = df["Score1"].shift(1, fill_value=df["Score1"].head(1)) != df["Score1"]
    df["Score2_chnage"] = df["Score2"].shift(1, fill_value=df["Score2"].head(1)) != df["Score2"]


    timestamp_for_score1=df.query('Score1_chnage == True')['Timestamp']
    timestamp_for_score2=df.query('Score2_chnage == True')['Timestamp']


    for timestamp1, timestamp2 in zip(timestamp_for_score1, timestamp_for_score2):
        end_time1.append(timestamp1)
        end_time2.append(timestamp2)


    for i,j in zip(end_time1, end_time2):
        start_time1.append(i-30.0)
        start_time2.append(j-30.0)
        


    final_time = start_time1 +start_time2 + end_time1 + end_time2
    final_time.sort()


        
    clip1 = clip.subclip(final_time[0], final_time[1])
    clip2 = clip.subclip(final_time[2], final_time[3])
    clip3 = clip.subclip(final_time[4], final_time[5])
    clip4 = clip.subclip(final_time[6], final_time[7])
    clip5 = clip.subclip(final_time[8], final_time[9])
    clip6 = clip.subclip(final_time[10], final_time[11])

    final = concatenate_videoclips([clip1, clip2, clip3, clip4,clip5, clip6])
    
    final.write_videofile("highlights.mp4")




match_video = 'D:\\Vison_projects\\Football_Highlights\\Full_match.mp4'
generate_highlights(match_video)