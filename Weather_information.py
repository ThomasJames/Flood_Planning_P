from tkinter import *
import requests

""""
Takes the user location and displayes the weather condition using a weather open weather api
""""


# https://www.youtube.com/watch?v=r9ZeTBsbMQo


def weather():
    url = "https://openweathermap.org/data/2...."


window = Tk()
window.geometry( "400x350" )

city_name_list = ["cowes"]

city_listbox = StringVar( window )
city_listbox.set( "select the city" )
option = OptionMenu( window, city_listbox, city_name_list )
option.grid( row=2, column=2, padx=150, pady=10 )

b1 = Button( window, text='o', width=15, command=weather )
b.grid( row=5, column=2, padx= )
