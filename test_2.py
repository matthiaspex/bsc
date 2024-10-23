string = "something.jpg"
if string.endswith(".jpg"):
    string = string[:-4]+".mp4"
print(string)