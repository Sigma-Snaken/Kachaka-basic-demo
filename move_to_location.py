from kachaka_api import KachakaApiClient
from kachaka_api.generated import kachaka_api_pb2 as pb2

sigma4 = KachakaApiClient(target="192.168.50.102:26400")
sigma4.set_speaker_volume(10)
print(sigma4.get_battery_info())

while True:
    print(sigma4.move_to_location("L01"))
    print(sigma4.speak("Hi jinjin"))
    print(sigma4.move_to_location("L02"))
    print(sigma4.speak("Hi papa"))
    print(sigma4.move_to_location("L03"))
    print(sigma4.speak("Hi mama"))