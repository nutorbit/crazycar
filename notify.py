import requests

url = "https://notify-api.line.me/api/notify"

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": "Bearer MCM5ChitwusEYHXMeXFczqQqtNeC1otUgYRyBwQSHHe"
}


def alert(message="test"):
    _ = requests.post(url=url, data={"message": message}, headers=headers)


if __name__ == "__main__":
    alert()
