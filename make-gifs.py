from moviepy.editor import *
clip = (VideoFileClip("/home/deval/synthetic-data/gifs/fake/cl26_act2.mp4"))
clip.write_gif("/home/deval/synthetic-data/gifs/fake/cl26_act2.gif",program='ffmpeg')