from tkinter import *
import requests

""""
Takes the user location and displayes the weather condition using a weather open weather api
""""


# https://www.youtube.com/watch?v=r9ZeTBsbMQo


def weather():
    city = city.listbox.get()
    url = "https://openweathermap.org/data/2...."
    res = requests.get( url )
    output = res.json()

    weahther_status = output['weather'][0]["description"]
    temperature = output['main']['temp']
    humidity = output["main"]["humidity"]
    wind_speed = output["wind"]["speed"]

    temperature_label.configure( text="weather status: ", str( temperature ) )
    humidity_label.configure( text="weather status: ", str( humidity ) )
    wind_speed_label.configure( text="weather status: ", str( wind_speed ) )
    weather_status_label.configure( text="weather status: ", str( weahther_status ) )


window = Tk()
window.geometry( "400x350" )

city_name_list = ["cowes"]

city_listbox = StringVar( window )
city_listbox.set( "select the city" )
option = OptionMenu( window, city_listbox, city_name_list )
option.grid( row=2, column=2, padx=150, pady=10 )

weather_status_label = Label( window, font=("times", 10, 'bold') )
weather_status_label.grid( rows=10, column=2 )

b1 = Button( window, text='o', width=15, command=weather )
b.grid( row=5, column=2, padx=150 )

temperature_label = Label( window, font=("times", 10, 'bold') )
temperature_label.grid( rows=10, column=2 )

humidity_label = Label( window, font=("times", 10, 'bold') )
humidity_label.grid( rows=10, column=2 )

wind_speed_label = Label( window, font=("times", 10, 'bold') )
wind_speed_label.grid( rows=10, column=2 )

window.mainloop()
