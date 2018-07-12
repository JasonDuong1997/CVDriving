import win32api as w32api
import time

keyList = ["\b"]

for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'$/\\":
	keyList.append(char)

def KeyCheck():
	keys = []
	for key in keyList:
		if w32api.GetAsyncKeyState(ord(key)):
			keys.append(key)
	return keys
