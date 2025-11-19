import requests

#url = 'http://localhost:9696/predict'
url = 'https://silent-thunder-9151.fly.dev/predict'

profile = {
    "parents": "great_pret",
    "has_nurs": "improper",
    "form": "completed",
    "children": "2",
    "housing": "less_conv",
    "finance": "inconv",
    "social": "problematic",
    "health": "priority"
}

response = requests.post(url, json=profile)
predictions = response.json()

print('response:', predictions)

n = predictions

if n == 0:
    print('This application is not_recommeded')
elif n == 1:
    print('This application is priority')
elif n == 2:
    print('This application is recommended')
elif n == 3:
    print('This application is Special_priority')
else:
    print('This application is very_recommeded')
