import win32api as w32api
import time

keyList = ["\b"]

for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'$/\\":
	keyList.append(char)

def keyCheck():
	keys = []
	for key in keyList:
		if w32api.GetAsyncKeyState(ord(key)):
			keys.append(key)
	return keys

# example of how keylogger works
while(True):
	if (keyCheck() == ["Q"]):
		break