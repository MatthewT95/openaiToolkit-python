import winsound

def main(request):
    winsound.Beep(int(request["frequency"]) or 1000, int(request["duration"]) or 1000)
    return "Beep successful."