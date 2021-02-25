"""
Data collection of Irish Folklore songs from: https://www.irishsongs.com/lyrics.php

"""
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import csv
import re

# website link
link = 'https://www.irishsongs.com/lyrics.php' # link to the main page
lyrics_links = 'https://www.irishsongs.com/' # base link for the songs

# the below will be uncommented at the end


def get_soup(link):

    # function: with link as input
    # :return: soup
    
    req = Request(link)
    response = urlopen(req)
    # print(response.status)
    soup = BeautifulSoup(response.read(), 'html.parser')
    return(soup)


# get song lyrics
def get_lyrics(new_link):
    """

    :param link: song link
    :return: str of song lyrics after cleaning
    """
    soup = get_soup(new_link)
    # clean and strip text from <p> tag
    lines = [line.strip() for line in str(soup.find('p')).split('<br/>')]
    new_lines = [] # to contain new lines after cleaning bad charachters
    bad_chars = ['\x92', '\x93', '\x94', '\x95', '\x96', '\x97']
    p_tag1, p_tag2 = '<p>', '</p>'
    for line in lines:
        # strip bad charachters and <p> tags
        line = ''.join([char for char in line if char not in bad_chars]).lower().replace('chorus','')
        line = re.sub('<[^<]+?>', '', line) # replace html tags with empty string
        if (line=='-') or (line==':') or (line==''): # left over from replacing chorus in some songs
            pass

        else:
            new_lines.append(line)
    return '\n'.join(new_lines)
# get all song a tags from main page
def get_song_dict(base_link = link):
    """

    :param base_link: takes in the soup html
    :return: song_dict: a dictionary with song_name, song_link and its lyrics
    """
    # songs dict
    songs_dict = {}
    main_soup = get_soup(base_link)
    # getting first/last songs
    first_song_tag = [tag for tag in main_soup.find_all('a') if tag.text=='A B C'][0]
    last_song_tag = [tag for tag in main_soup.find_all('a') if tag.text=='Zebra Roses'][0]
    # getting all a tags
    a_tags = main_soup.find_all('a')

    # get all song tags between first/last song indices
    first_song_index = a_tags.index(first_song_tag)
    last_song_index = a_tags.index(last_song_tag)

    for tag in a_tags[first_song_index:last_song_index+1]:
        song_name = tag.text
        song_link = lyrics_links+(tag['href'])

        lyrics = get_lyrics(song_link)
        songs_dict[song_name] = [song_link, lyrics]



    return songs_dict


songs_dict = get_song_dict(link)

# writing the dictionary to csv file
with open('data/songs_csv.csv', 'w') as csvfile:
    fieldnames = ['song_name', 'link', 'lyrics']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for key, value in songs_dict.items():
        writer.writerow({fieldnames[0]: key,
                         fieldnames[1]: value[0],
                         fieldnames[2]: value[1]})
